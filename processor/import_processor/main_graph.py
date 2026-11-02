"""
导入功能的主流程图
"""
import json
import logging

from langgraph.graph import StateGraph
from langgraph.constants import START, END

from processor.import_processor.base import setup_logging
from processor.import_processor.state import ImportGraphState
from processor.import_processor.nodes.node_entry import NodeEntry
from processor.import_processor.nodes.node_pdf_to_md import NodePDFToMD
from processor.import_processor.nodes.node_md_img import NodeMDImg
from processor.import_processor.nodes.node_document_split import NodeDocumentSplit
from processor.import_processor.nodes.node_item_name_recognition import NodeItemNameRecognition
from processor.import_processor.nodes.node_bge_embedding import NodeBgeEmbedding
from processor.import_processor.nodes.node_import_milvus import NodeImportMilvus

class KBImportWorkflow:

    def __init__(self):
        """
        初始化工作流对象
        """
        self._compiled_graph = None

    @property
    def graph(self):
        """
        延迟加载（懒加载）：只在第一次使用工作流对象的时候编译图
        :return:
        """
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph()
        return self._compiled_graph

    @staticmethod
    def route_after_entry(state: ImportGraphState):

        if state.get("is_pdf_read_enabled"):
            return "node_pdf_to_md"
        elif state.get("is_md_read_enabled"):
            # return "node_md_img"
            return "node_document_split"
        else:
            return END


    def build_graph(self):
        """
        创建图结构
        :return: 编译后的图
        """
        # 1. 初始化图(工作流)
        graph = StateGraph(ImportGraphState)

        # 2. 在图中注册节点
        graph.add_node("node_entry", NodeEntry())
        graph.add_node("node_pdf_to_md", NodePDFToMD())
        graph.add_node("node_md_img", NodeMDImg())
        graph.add_node("node_document_split", NodeDocumentSplit())
        graph.add_node("node_item_name_recognition", NodeItemNameRecognition())
        graph.add_node("node_bge_embedding", NodeBgeEmbedding())
        graph.add_node("node_import_milvus", NodeImportMilvus())

        # 3. 设置入口节点
        graph.set_entry_point("node_entry")

        # 4. 注册条件边
        graph.add_conditional_edges(
            "node_entry",
            self.route_after_entry,
            {
                # key: 路由函数的返回值，value: 节点的名字
                "node_md_img": "node_md_img",
                "node_pdf_to_md": "node_pdf_to_md",

                END: END
            }
        )

        # 5. 注册顺序边
        graph.add_edge("node_pdf_to_md", "node_md_img")
        graph.add_edge("node_md_img", "node_document_split")
        graph.add_edge("node_document_split", "node_item_name_recognition")
        graph.add_edge("node_item_name_recognition", "node_bge_embedding")
        graph.add_edge("node_bge_embedding", "node_import_milvus")
        graph.add_edge("node_import_milvus", END)

        # 6. 编译工作流
        return graph.compile()



    def run(self, state: ImportGraphState, stream:bool=False):
        """
        统一执行入口，支持切换invoke/stream
        :param state:初始状态
        :param stream:是否流式输出
        :return:结果
        """

        if stream:
            return self.graph.stream(state)
        else:
            return self.graph.invoke(state)

if __name__ == '__main__':
    # 激活全局日志
    setup_logging()
    # 定义初始状态
    init_state = {
        "import_file_path": r"D:\python\knowledge_base\doc\hak180使用说明书.pdf",
        "file_dir": r"D:/python/knowledge_base/output"
    }


    workflow = KBImportWorkflow()


    final_state = workflow.run(init_state, stream=False)
    logging.getLogger().info(final_state)

    # 打印编译后的图结构
    # workflow.graph.get_graph().print_ascii()

