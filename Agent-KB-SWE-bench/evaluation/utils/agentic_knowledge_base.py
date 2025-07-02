import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class WorkflowStep:
    step_id: int = -1
    act_command: str = ''
    act_background: str = ''
    act_thought: str = ''
    act_message: str = ''
    obv_observation: str = ''
    obv_content: str = ''
    obv_extras: Dict[str, Any] = field(default_factory=dict)  # 使用 default_factory
    obv_message: str = ''
    Structured_Triplets: tuple = ()
    raw_text: str = ''
    LLM_Summarized_Text: str = ''


@dataclass
class AgentInteraction:
    sender_id: str
    receiver_id: str
    message_type: str  # e.g., "request", "response", "error"
    content: Dict[str, Any]
    timestamp: datetime = datetime.now()


@dataclass
class WorkflowInstance:
    """一个完整的工作流实例"""

    workflow_id: str
    workflow_metadata: Dict[str, Any] = None  # 实例元数据
    resolved: bool = False  # workflow是否成功
    instruction: str = None  # 工作流的指令
    model_output: str = None  # 模型的输出

    # created_at: datetime = datetime.now()
    steps: Dict[str, WorkflowStep] = None  # 步骤ID到步骤对象的映射
    # dependencies: nx.DiGraph = None  # 步骤依赖关系图
    # context: Dict[str, Any] = None  # 工作流的共享上下文（跨步骤数据）
    interactions: List[AgentInteraction] = None  # 添加到类中

    def update_context(self, new_data: Dict[str, Any]):
        self.context_history.append(self.context.copy())
        self.context.update(new_data)

    def rollback_context(self, steps_back: int = 1):
        if len(self.context_history) >= steps_back:
            self.context = self.context_history[-steps_back]


class AgenticKnowledgeBase:
    """
    Agentic Knowledge Base (AKB) class for managing agentic knowledge.
    """

    def __init__(self, all_data_path_l=None, eval_report_dir_l=None):
        # 核心存储
        self.workflows: Dict[str, WorkflowInstance] = {}  # ID到实例
        for all_data_path in all_data_path_l:
            if not os.path.exists(all_data_path):
                raise FileNotFoundError(
                    f'[ERROR] all_data_path: {all_data_path} does not exist.'
                )
            print(f'[INFO] Parsing workflow instances from {all_data_path}')
            # 解析工作流实例
            self.parse_workflow_instance(all_data_path=all_data_path)

        # 索引结构（加速查询）
        # self.agent_workflow_index: Dict[str, List[str]] = {}  # 智能体ID参与的流程
        # self.status_index: Dict[str, List[str]] = {}  # 状态到流程ID
        # self.timestamp_index: List[tuple] = []  # 时间排序的流程ID

    def traverse_files(self, directory_path):
        all_reports_dict = defaultdict(str)
        # 遍历指定目录下的所有文件和子文件夹
        for root, dirs, files in os.walk(directory_path):
            for file_name in files:
                if file_name.endswith('.json'):
                    # 拼接文件路径
                    file_path = os.path.join(root, file_name)
                    # 读取文件内容
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        all_reports_dict[root.split('/')[-1]] = data

        return all_reports_dict

    def parse_workflow_instance(self, all_data_path, eval_report_dir=None):
        # all_reports_dict = self.traverse_files(eval_report_dir)
        # print(f'[INFO] all_reports_dict: {len(all_reports_dict)}, {all_reports_dict.keys()}')
        all_data = {}
        with open(all_data_path, 'r') as f:
            for line in f:
                data = json.loads(
                    line
                )  # dict_keys(['instance_id', 'swe_instance', 'instruction', 'git_patch', 'metadata', 'history', 'test_result'])

                all_data[data['instance_id']] = data

        for workflow_id, data in all_data.items():
            try:
                # reports = all_reports_dict[workflow_id][workflow_id] # 格式化的测试用例等test信息，内容和data['test_result']一致
                # resolved = reports['resolved']
                # print(data['report'])
                resolved = data['report']['resolved']  # 是否修复bug成功

                # if resolved:
                print(f'[INFO] processing workflow: {workflow_id}')
                print(data.keys())
                # parse data
                instruction = data['instruction']
                git_patch = data['git_patch']  # 模型生成的patch
                workflow_metadata = data['metadata']
                # result = data['test_result']['result'] # 模型生成的测试结果，比report.json多了test_timeout和test_errored字段
                test_metadata = data['test_result'][
                    'metadata'
                ]  # 包含了测试流程的每个环节的状态
                # test_output = data['test_result']['test_output'] # 模型结果的测试log，和report.json中的数据格式化后是一样的
                # result_raw = data['test_result']['result_raw'] # 模型生成的测试结果，和report.json中的数据是一样的

                workflow_instance = WorkflowInstance(
                    workflow_id=workflow_id,
                    workflow_metadata=workflow_metadata,
                    resolved=resolved,
                    instruction=instruction,
                    model_output=git_patch,
                )

                workflow_steps_l = []
                for step_idx, record in enumerate(data['history']):
                    assert (
                        len(record) == 2
                    ), f'Expected 2 elements in record, got {len(record)}'
                    action_record, observation_record = record
                    act_command = action_record['args'].get(
                        'command', None
                    )  # action的指令
                    act_background = action_record['args'].get('background', None)  #
                    act_thought = action_record['args'].get(
                        'thought', None
                    )  # 该step思考的内容
                    act_message = action_record[
                        'message'
                    ]  # 执行的动作，和act_command内容一致
                    obv_observation = observation_record[
                        'observation'
                    ]  # run/message/null
                    obv_content = observation_record['content']  # 观察repo的内容
                    obv_extras = observation_record[
                        'extras'
                    ]  # 额外的信息，基本没用。和obv_message是一样的内容，只是一个格式化dict的形式
                    obv_message = observation_record[
                        'message'
                    ]  # 和ovc_extras一样的内容，string

                    workflow_step = WorkflowStep(
                        step_id=step_idx,
                        act_command=act_command,
                        act_background=act_background,
                        act_thought=act_thought,
                        act_message=act_message,
                        obv_observation=obv_observation,
                        obv_content=obv_content,
                        obv_extras=obv_extras,
                        obv_message=obv_message,
                    )

                    workflow_steps_l.append(workflow_step)

                workflow_instance.steps = workflow_steps_l
                self.add_workflow_instance(workflow_instance)
                # else:
                #     print(f'[INFO] UnResolved workflow: {workflow_id}')

            except Exception as e:
                print(f'[ERROR] {e}')
                continue
        print('[INFO] Finished parsing workflow instances.')
        return

    def add_workflow_instance(self, workflow: WorkflowInstance):
        """增加新的工作流实例"""
        self.workflows[workflow.workflow_id] = workflow
        return workflow

    # def __getitem__(self, key):
    #     return self.kb[key]

    # def __setitem__(self, key, value):
    #     self.kb[key] = value

    # def get(self, key, default=None):
    #     return self.kb.get(key, default)

    # def delete(self, key, default=None):
    #     return self.kb.pop(key, default)

    # def add_step_dependency(self, workflow_id: str, from_step: str, to_step: str):
    #     """添加步骤间的依赖关系"""
    #     workflow = self.workflows[workflow_id]
    #     workflow.dependencies.add_edge(from_step, to_step)

    # def get_execution_order(self, workflow_id: str) -> List[str]:
    #     """获取拓扑排序后的执行顺序"""
    #     workflow = self.workflows[workflow_id]
    #     return list(nx.topological_sort(workflow.dependencies))

    # def search_workflows(self,
    #                      status: Optional[str] = None,
    #                      agent_id: Optional[str] = None,
    #                      created_after: Optional[datetime] = None) -> List[WorkflowInstance]:
    #     """多条件组合查询"""
    #     results = []
    #     # 利用索引加速
    #     if agent_id:
    #         candidate_ids = self.agent_workflow_index.get(agent_id, [])
    #     else:
    #         candidate_ids = self.workflows.keys()

    #     for wf_id in candidate_ids:
    #         wf = self.workflows[wf_id]
    #         if status and wf.current_status != status:
    #             continue
    #         if created_after and wf.created_at < created_after:
    #             continue
    #         results.append(wf)
    #     return results


class AKB_Manager:
    """
    Agentic Knowledge Base (AKB) class for managing agentic knowledge.
    """

    def __init__(self, all_data_path_l):
        """
        Initialize the AKB with an empty knowledge base.
        """
        all_data_path_l = all_data_path_l
        # [
        #     './evaluation/benchmarks/swe_bench/examples/example_agent_output.jsonl'
        # ]
        eval_report_dir_l = [
            './evaluation/benchmarks/swe_bench/examples/eval_outputs_20250405_175943'
        ]
        self.knowledge_base = AgenticKnowledgeBase(
            all_data_path_l=all_data_path_l, eval_report_dir_l=eval_report_dir_l
        )

    def add_knowledge(self, key, value):
        """
        Add knowledge to the AKB.

        :param key: The key for the knowledge.
        :param value: The value of the knowledge.
        """
        self.knowledge_base[key] = value

    def get_knowledge(self, key):
        """
        Retrieve knowledge from the AKB.

        :param key: The key for the knowledge.
        :return: The value of the knowledge or None if not found.
        """
        return self.knowledge_base.get(key, None)

    def get_all_workflow(self):
        """
        Retrieve all workflows from the AKB.

        :return: A list of all workflow instances.
        """
        return self.knowledge_base.workflows

    def delete_knowledge(self, key):
        """
        Retrieve knowledge from the AKB.

        :param key: The key for the knowledge.
        :return: The value of the knowledge or None if not found.
        """
        return self.knowledge_base.delete(key, None)
