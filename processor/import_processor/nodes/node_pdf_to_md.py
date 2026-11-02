import shutil
import time
from pathlib import Path
import zipfile
import requests
from pypdf import PdfReader, PdfWriter

from config.mineru_config import mineru_config
from processor.import_processor.base import BaseNode, T, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, ConfigurationError, \
    PdfConversionError
from processor.import_processor.state import ImportGraphState

# MinerU 接口限制：单个 PDF 文件最多 200 页，超出会返回
# "number of pages exceeds limit (200 pages), please split the file and try again"
MINERU_MAX_PAGES: int = 200


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """
    name = "node_pdf_to_md"

    def process(self, state: T) -> T:
        """

        :param state:
        :return:
        """
        # 1. 参数校验并返回path结果
        pdf_path_obj, output_dir_obj, = self._step_1_validate_path(state)

        # 2. 预处理PDF：空密码解密 + 超过200页自动拆分（MinerU单文件页数上限200）
        pdf_parts = self._step_1_5_prepare_pdf(pdf_path_obj, output_dir_obj)

        # 3. 将每个分片上传到Mineru并轮训结果
        zip_urls = [self._step_2_upload_and_poll(part) for part in pdf_parts]

        # 4. 下载ZIP包、解压并按分片顺序合并MD
        md_path = self._step_3_download_and_extract(zip_urls, output_dir_obj, pdf_path_obj.stem)

        # 4. 将MD文件的内容读取出来
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 5. 返回结果
        state["md_content"] = md_content
        state["md_path"] = str(md_path)
        return state

    def _step_1_validate_path(self, state: ImportGraphState):
        """
        步骤1：校验PDF文件路径和输出目录
        核心职责：参数非空校验 | 路径转换 | PDF文件有效性校验 | 输出目录自动创建
        返回：合法的PDF文件Path对象、输出目录Path对象
        异常：ValueError(参数缺失)、FileNotFoundError(文件无效)
        :param state:
        :return:
        """
        # 1. 参数得到非空校验
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(field_name='pdf_path', message="PDF路径不能为空", expected_type=str)

        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(field_name='file_dir', message="输出路径不能为空", expected_type=str)

        # 2. 转换为Path对象
        pdf_path_obj = Path(pdf_path)
        file_dir_obj = Path(file_dir)

        # 3. pdf是否存在
        if not pdf_path_obj.exists():
            raise FileProcessingError(f"文件{pdf_path_obj.name}不存在")

        # 4. 输出目录不存在则创建
        if not file_dir_obj.exists():
            self.logger.info(f"输出目录{file_dir_obj.absolute()}不存在，正在创建")
            file_dir_obj.mkdir(parents=True, exist_ok=True)

        return pdf_path_obj, file_dir_obj

    def _step_1_5_prepare_pdf(self, pdf_path_obj: Path, output_dir_obj: Path) -> list[Path]:
        """
        步骤1.5：上传前的PDF预处理
        处理两类会导致 MinerU 解析失败的文件：
            1. 带权限加密（/Encrypt）的PDF：MinerU 只会返回含糊的 "retry limit reached"，
               若用户密码为空则用空密码解密后重新写出一份无加密的PDF。
            2. 页数超过200页的PDF：MinerU 单文件上限200页，按200页拆分成多个分片依次解析。
        返回：需要上传的PDF文件列表（无需处理时就是原文件）
        """
        try:
            reader = PdfReader(str(pdf_path_obj))
        except Exception as e:
            raise PdfConversionError(f"PDF文件无法读取，请确认文件是否损坏：{e}")

        # 1. 加密处理：空用户密码可解密，否则必须让用户先自行解密
        was_encrypted = reader.is_encrypted
        if was_encrypted:
            try:
                # decrypt 返回0表示失败（需要真正的用户打开密码）
                if not reader.decrypt(""):
                    raise PdfConversionError(
                        f"PDF文件已加密且需要打开密码，MinerU无法解析，请先解密后再上传：{pdf_path_obj.name}")
            except PdfConversionError:
                raise
            except Exception as e:
                raise PdfConversionError(f"PDF解密失败，请先解密后再上传：{pdf_path_obj.name}，原因：{e}")

        total_pages = len(reader.pages)

        # 2. 无需处理：未加密且页数未超限，直接上传原文件
        if not was_encrypted and total_pages <= MINERU_MAX_PAGES:
            return [pdf_path_obj]

        # 3. 需要重写：去加密和/或按200页拆分
        parts: list[Path] = []
        for start in range(0, total_pages, MINERU_MAX_PAGES):
            writer = PdfWriter()
            for page in reader.pages[start:start + MINERU_MAX_PAGES]:
                writer.add_page(page)

            part_path = output_dir_obj / f"{pdf_path_obj.stem}_part{len(parts) + 1}.pdf"
            with open(part_path, "wb") as f:
                writer.write(f)
            parts.append(part_path)

        self.logger.info(
            f"【PDF预处理】共{total_pages}页，加密={was_encrypted}，"
            f"已拆分为{len(parts)}个分片（每片最多{MINERU_MAX_PAGES}页）")
        return parts

    def _step_2_upload_and_poll(self, pdf_path_obj: Path):
        """
        步骤2：上传PDF至MinerU并轮询解析任务状态
        核心流程：配置校验 → 获取上传链接 → 文件上传 → 任务轮询（直至完成/失败/超时）
        参数：pdf_path_obj-已校验的PDF Path对象
        返回：解析结果ZIP包下载链接full_zip_url
        异常：ValueError(配置缺失)、RuntimeError(请求/上传失败)、TimeoutError(任务超时)
        """
        # 1、配置文件校验
        if not self.config.mineru_base_url:
            raise ConfigurationError(message="MinerU配置缺失：请在 .env 文件中正确配置 MINERU_BASE_URL 参数")
        if not self.config.mineru_api_key:
            raise ConfigurationError(message="MinerU配置缺失：请在 .env 文件中正确配置 MINERU_API_KEY 参数")

        # 2、从MinerU服务器获取上传链接
        token = self.config.mineru_api_key
        url = f"{self.config.mineru_base_url}/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }

        # 获取上传url和任务的batch_id
        response = requests.post(url, headers=header, json=data)
        # 对响应结果进行校验
        # 先校验http状态
        if response.status_code != 200:
            raise PdfConversionError(message=f"获取上传链接响应失败：状态码：{response.status_code}，响应结果：{response}")
        # 校验业务码
        result = response.json()
        if result.get("code") != 0:
            raise PdfConversionError(f"获取上传链接失败：返回数据：{result}")

            # 获取响应结果
        signed_url = result["data"]["file_urls"][0]
        batch_id = result["data"]["batch_id"]

        # 3、文件上传
        with open(pdf_path_obj, "rb") as f:
            res_upload = requests.put(signed_url, data=f)
            if res_upload.status_code != 200:
                raise PdfConversionError(f"文件上传失败：状态码：{res_upload.status_code}，响应结果：{res_upload}")

            self.logger.info(f"文件上传成功！")

        # 4、批量获取任务结果
        poll_url = f"{mineru_config.mineru_base_url}/extract-results/batch/{batch_id}"

        # 轮询查询
        start_time = time.time()
        timeout_seconds = 600
        poll_interval = 3
        self.logger.info(f"【任务轮询】最大超时：{timeout_seconds}s，batch_id：{batch_id}")

        while True:
            # 已消耗的时间
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise TimeoutError(f"【任务轮询】超时！任务处理超{timeout_seconds}秒，batch_id：{batch_id}")

            # 获取结果
            try:
                poll_res = requests.get(poll_url, headers=header, timeout=10)
            except Exception as e:
                self.logger.warning(f"【任务轮询】网络请求异常，{poll_interval}秒后重试：{str(e)}，batch_id：{batch_id}")
                time.sleep(poll_interval)
                continue
            # 处理HTTP响应错误
            if poll_res.status_code != 200:
                raise PdfConversionError(f"【任务轮询】HTTP请求失败，状态码：{poll_res.status_code}，响应内容：{poll_res}")
            # 解析轮询结果，校验业务状态
            poll_data = poll_res.json()
            if poll_data["code"] != 0:
                raise PdfConversionError(f"【任务轮询】业务错误，返回数据：{poll_data}")

            extract_results = poll_data["data"]["extract_result"]

            # 获取结果
            result_item = extract_results[0]
            data_state = result_item["state"]

            # 状态为 done
            if data_state == "done":
                self.logger.info(f"【任务轮询】解析任务完成！总耗时{int(elapsed_time)}s，bactch_id：{batch_id}")

                full_zip_url = result_item["full_zip_url"]
                self.logger.info(f"【任务轮询】返回ZIP包下载链接：{full_zip_url}，bactch_id：{batch_id}")

                return full_zip_url

            elif data_state == "failed":
                err_msg = result_item.get("err_msg", "未知错误，无具体信息")
                raise PdfConversionError(f"【任务轮询】解析任务失败！batch_id：{batch_id}，错误信息：{err_msg}")

            else:
                self.logger.info(
                    f"【任务轮询】处理中... 已耗时{int(elapsed_time)}s，状态：{data_state}， batch_id：{batch_id}")
                time.sleep(poll_interval)

    def _step_3_download_and_extract(self, zip_urls: list[str], output_dir_obj: Path, pdf_stem: str) -> str:
        """
        步骤3：下载MinerU解析结果ZIP包并解压，按分片顺序合并MD
        核心流程：下载ZIP → 每个分片解压到独立临时目录 → 收集full.md → 合并其余产物 → 按顺序拼接MD
        参数：zip_urls-各分片的ZIP下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
        返回：最终MD文件的字符串格式绝对路径
        异常：FileProcessingError(下载失败/无MD文件)
        """
        # 1. 清理旧的解压目录（重跑时保证干净）
        extract_target_dir = output_dir_obj / pdf_stem
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir)
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        md_contents: list[str] = []

        for idx, zip_url in enumerate(zip_urls, start=1):
            # 2. 下载zip
            response = requests.get(zip_url, timeout=300)
            if response.status_code != 200:
                raise FileProcessingError(f"第{idx}/{len(zip_urls)}个分片下载失败，状态码：{response.status_code}")

            # 单分片沿用原来的命名，多分片按 part 编号区分
            if len(zip_urls) == 1:
                zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
            else:
                zip_save_path = output_dir_obj / f"{pdf_stem}_result_part{idx}.zip"
            with open(zip_save_path, "wb") as f:
                f.write(response.content)
            self.log_step("ZIP下载", f"第{idx}/{len(zip_urls)}个分片下载成功")

            # 3. 解压到独立临时目录，避免多个分片的 full.md 互相覆盖
            part_dir = extract_target_dir / f"_part{idx}"
            part_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"【ZIP解压】开始解压第{idx}/{len(zip_urls)}个分片 ...")
            with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
                zip_file_obj.extractall(part_dir)

            # 4. 收集 full.md
            full_md_path = part_dir / "full.md"
            if not full_md_path.exists():
                raise FileProcessingError(f"第{idx}个分片未找到 full.md，ZIP内容异常")
            md_contents.append(full_md_path.read_text(encoding="utf-8"))

            # 5. 把除 full.md 外的产物（images/json/origin.pdf等）合并到目标目录
            for item in part_dir.iterdir():
                if item.name == "full.md":
                    continue
                target = extract_target_dir / item.name
                if item.name == "images" and target.exists():
                    # 图片按内容哈希命名，同名即同内容，直接合并
                    self._merge_dir(item, target)
                    shutil.rmtree(item)
                else:
                    if target.exists():
                        if target.is_dir():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                    shutil.move(str(item), str(target))
            shutil.rmtree(part_dir)

        # 6. 按分片顺序合并写入最终MD
        new_md_path = extract_target_dir / f"{pdf_stem}.md"
        with open(new_md_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(md_contents))
        self.logger.info(f"【MD合并】共{len(md_contents)}个分片，已合并为：{new_md_path}")

        return str(new_md_path.absolute())

    @staticmethod
    def _merge_dir(src: Path, dst: Path) -> None:
        """把 src 目录下的文件移动到 dst；同名文件视为内容相同，跳过。"""
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.iterdir():
            target = dst / p.name
            if target.exists():
                continue
            shutil.move(str(p), str(target))


if __name__ == '__main__':
    # 激活日志
    setup_logging()

    init_state = {
        "pdf_path": r"D:\python\knowledge_base\output\成都景点推荐.md",
        "file_dir": r"D:/python/knowledge_base/output"
    }

    node_pdf_to_md = NodePDFToMD()

    result = node_pdf_to_md(init_state)


