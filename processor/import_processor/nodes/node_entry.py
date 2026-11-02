import json
import logging
from pathlib import Path
from processor.import_processor.base import BaseNode, T, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, ValidationError
from processor.import_processor.state import ImportGraphState

class NodeEntry(BaseNode):
    """
    入口节点：任务分发
    """
    name = "node_entry"

    def process(self, state: T) -> T:
        """
        判断文件的扩展名是pdf还是md
        :param state:
        :return:
        """
        # 1. 从state中获取文件的绝对路径
        import_file_path = state.get("import_file_path")

        # 判断路径是否为空
        if not import_file_path:
            raise StateFieldError(
                node_name="node_entey",
                field_name="import_file_path",
                message="文件路径不能为空",
                expected_type=str,

            )
        # 2. 将import_file_path转换维Path对象
        import_file_path_obj = Path(import_file_path)
        if not import_file_path_obj:
            raise FileProcessingError(message=f"文件{import_file_path_obj.name}不存在")

        # 3. 检查文件的后缀
        if import_file_path_obj.suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        elif import_file_path_obj.suffix == ".md":
            with open(import_file_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            state["md_content"] = md_content
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        else:
            raise ValidationError(message=f"文件{import_file_path_obj.suffix}不支持")

        # 4. 获取上传文件的标题，更新到state中
        state["file_title"] = import_file_path_obj.stem


        return state

if __name__ == '__main__':
    # 激活全局日志
    setup_logging()
    init_state = {
        "import_file_path": r"D:\python\knowledge_base\output\成都景点推荐.md"
    }

    node_entry = NodeEntry()
    result = node_entry(init_state)
    json_state = json.dumps(result, ensure_ascii=False, indent=4)

    logging.getLogger().info(result)











