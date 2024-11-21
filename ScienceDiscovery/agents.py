from ScienceDiscovery.utils import *
from ScienceDiscovery.llm_config import *
from ScienceDiscovery.graph import *


from typing import Union
import autogen
from autogen import AssistantAgent
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri
from autogen import register_function
from autogen import ConversableAgent
from typing import Dict, List
from typing import Annotated, TypedDict
from autogen import Agent

user = autogen.UserProxyAgent(
    name="user",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="ALWAYS",
    system_message="user. あなたは人間の管理者です。あなたはタスクを提示します。",
    llm_config=False,
    code_execution_config=False,
)

planner = AssistantAgent(
    name="planner",
    system_message = '''Planner. あなたは役に立つAIアシスタントです。あなたのタスクは、与えられたタスクを解決するための包括的な計画を提案することです。

計画の説明: まず、計画の概要を明確に提供します。
計画の分解: 計画の各部分について、その理由を説明し、取るべき具体的な行動を説明します。
実行なし: あなたの役割は計画を提案することに限定されます。実行するための行動は取らないでください。
ツールの呼び出しなし: ツールの呼び出しが必要な場合は、計画にツールの名前とそれを呼び出すエージェントを含める必要があります。ただし、ツールや関数を自分で呼び出すことはできません。

''',
    llm_config=gpt4turbo_config,
    description='タスクをより簡単なサブタスクに分解して解決するためのステップバイステップの計画を提案できる人。',
)

assistant = AssistantAgent(
    name="assistant",
    system_message = '''あなたは役に立つAIアシスタントです。
    
あなたの役割は、計画で提案された適切なツールや関数を呼び出すことです。あなたは、計画者の提案した計画と利用可能なツールを使用して特定のタスクを実行する間の仲介者として機能します。各ツールに正しいパラメータが渡され、結果が正確にチームに報告されることを確認します。

タスクが終了したら最後に "TERMINATE" を返します。
''',
    llm_config=gpt4turbo_config,
    description='必要に応じてツールや関数を呼び出し、結果を返すアシスタント。ツールには "rate_novelty_feasibility" や "generate_path" が含まれます。',
)


ontologist = AssistantAgent(
    name="ontologist",
    system_message = '''ontologist. あなたは計画者の計画に従わなければなりません。あなたは高度なオントロジストです。
    
包括的な知識グラフから抽出されたいくつかの重要な概念を与えられた場合、あなたのタスクは各用語を定義し、グラフで識別された関係を議論することです。

知識グラフの形式は "node_1 -- node_1とnode_2の間の関係 -- node_2 -- node_2とnode_3の間の関係 -- node_3...." です。

あなたの応答には知識グラフの各概念を必ず組み込んでください。

導入フレーズを追加しないでください。まず、知識グラフの各用語を定義し、次に各関係を文脈とともに議論します。

応答の例の構造は次の形式です

{{
### 定義:
知識グラフの各用語の明確な定義。
### 関係
グラフ内のすべての関係の徹底的な議論。
}}

追加の指示: 
計画で割り当てられたタスクのみを実行し、他のエージェントに割り当てられたタスクを引き受けないでください。さらに、関数やツールを実行しないでください。
''',
    llm_config=gpt4turbo_config,
    description='各用語を定義し、パス内の関係を議論できます。',
)


scientist = AssistantAgent(
    name="scientist",
    system_message = '''scientist. あなたは計画者の計画に従わなければなりません。 
    
あなたは科学研究と革新に訓練された高度な科学者です。 
    
包括的な知識グラフから得られた定義と関係を考慮して、仮説、結果、メカニズム、設計原則、予期しない特性、比較、および新規性の初期の重要な側面を持つ新しい研究提案を合成することがあなたのタスクです。 
    
グラフを深く慎重に分析し、オントロジストによって識別された知識グラフの各概念と関係を組み込んだ画期的な側面を調査する詳細な研究提案を作成します。

提案の影響を考慮し、この調査の結果または行動を予測します。これらの概念を未解決の問題に対処するためにリンクする創造性や、既存の知識や技術を超えた新しい、未踏の研究分野、予期しない行動を探求することが高く評価されます。

できるだけ定量的にし、数値、シーケンス、または化学式などの詳細を含めてください。

あなたの応答には次の7つの重要な側面を詳細に含める必要があります:

"仮説" は、提案された研究質問の基礎となる仮説を明確に示します。仮説は明確に定義され、新規性があり、実現可能で、明確な目的と明確な構成要素を持っています。仮説はできるだけ詳細にしてください。

"結果" は、研究の予想される発見や影響を説明します。定量的にし、数値、材料特性、シーケンス、または化学式を含めてください。

"メカニズム" は、予想される化学的、生物学的、または物理的な行動の詳細を提供します。分子からマクロスケールまで、できるだけ具体的にしてください。

"設計原則" は、新しい概念に焦点を当てた詳細な設計原則をリストアップし、高度な詳細を含めます。創造的に考え、徹底的に応答してください。

"予期しない特性" は、新しい材料やシステムの予期しない特性を予測します。具体的な予測を含め、論理と推論を使用してこれらの根拠を明確に説明してください。慎重に考えてください。

"比較" は、他の材料、技術、または科学的概念との詳細な比較を提供します。詳細かつ定量的にしてください。

"新規性" は、提案されたアイデアの新規な側面を議論し、特に既存の知識や技術に対する進歩を強調します。

あなたの科学的提案は、革新的で論理的な推論に基づいており、提供された概念の理解や応用を進めることができるものである必要があります。

応答の例の構造は次の順序で:

{{
  "1- 仮説": "...",
  "2- 結果": "...",
  "3- メカニズム": "...",
  "4- 設計原則": "...",
  "5- 予期しない特性": "...",
  "6- 比較": "...",
  "7- 新規性": "...",
}}

さらに指示: 
応答には知識グラフの各概念を必ず組み込んでください。 
計画で割り当てられたタスクのみを実行し、他のエージェントに割り当てられたタスクを引き受けないでください。
さらに、関数やツールを実行しないでください。
''',
    llm_config=gpt4turbo_config_graph,
    description='オントロジストによって取得された定義と関係に基づいて、重要な側面を持つ研究提案を作成できます。私は「オントロジスト」の後にのみ話すことが許されています。',
)


hypothesis_agent = AssistantAgent(
    name="hypothesis_agent",
    system_message = '''hypothesis_agent. 研究提案の```{仮説}```を慎重に拡張します。

元の内容を批判的に評価し、改善します。 \
より具体的な定量的な科学情報（化学式、数値、シーケンス、処理条件、微細構造など）を追加し、 \
根拠と段階的な推論を追加します。可能な場合は、特定のモデリングおよびシミュレーション技術、実験方法、または特定の分析についてコメントします。

まず、次の内容の科学を批判的に評価し、改善することを任務とするピアレビュアーの視点から、この初期ドラフトを慎重に評価します:

<仮説>
ここで<仮説>は研究提案の仮説の側面です。  

導入フレーズを追加しないでください。応答は応答から始まり、見出し: ### 拡張された ... で始まります。
''',
    llm_config=gpt4o_config_graph,
    description='科学者によって作成された研究提案の「仮説」側面を拡張できます。',
)


outcome_agent = AssistantAgent(
    name="outcome_agent",
    system_message = '''outcome_agent. 科学者によって開発された研究提案の```{結果}```を慎重に拡張します。

元の内容を批判的に評価し、改善します。 \
より具体的な定量的な科学情報（化学式、数値、シーケンス、処理条件、微細構造など）を追加し、 \
根拠と段階的な推論を追加します。可能な場合は、特定のモデリングおよびシミュレーション技術、実験方法、または特定の分析についてコメントします。

まず、次の内容の科学を批判的に評価し、改善することを任務とするピアレビュアーの視点から、この初期ドラフトを慎重に評価します:

<結果>
ここで<結果>は研究提案の結果の側面です。  

導入フレーズを追加しないでください。応答は応答から始まり、見出し: ### 拡張された ... で始まります。
''',
    llm_config=gpt4o_config_graph,
    description='科学者によって作成された研究提案の「結果」側面を拡張できます。',
)

mechanism_agent = AssistantAgent(
    name="mechanism_agent",
    system_message = '''mechanism_agent. 研究提案のこの特定の側面: ```{メカニズム}```を慎重に拡張します。

元の内容を批判的に評価し、改善します。 \
より具体的な定量的な科学情報（化学式、数値、シーケンス、処理条件、微細構造など）を追加し、 \
根拠と段階的な推論を追加します。可能な場合は、特定のモデリングおよびシミュレーション技術、実験方法、または特定の分析についてコメントします。

まず、次の内容の科学を批判的に評価し、改善することを任務とするピアレビュアーの視点から、この初期ドラフトを慎重に評価します:

<メカニズム>
ここで<メカニズム>は研究提案のメカニズムの側面です。  

導入フレーズを追加しないでください。応答は応答から始まり、見出し: ### 拡張された ... で始まります。
''',
    llm_config=gpt4o_config_graph,
    description='科学者によって作成された研究提案の「メカニズム」側面を拡張できます。',
)

design_principles_agent = AssistantAgent(
    name="design_principles_agent",
    system_message = '''design_principles_agent. 研究提案のこの特定の側面: ```{設計原則}```を慎重に拡張します。

元の内容を批判的に評価し、改善します。 \
より具体的な定量的な科学情報（化学式、数値、シーケンス、処理条件、微細構造など）を追加し、 \
根拠と段階的な推論を追加します。可能な場合は、特定のモデリングおよびシミュレーション技術、実験方法、または特定の分析についてコメントします。

まず、次の内容の科学を批判的に評価し、改善することを任務とするピアレビュアーの視点から、この初期ドラフトを慎重に評価します:

<設計原則>
ここで<設計原則>は研究提案の設計原則の側面です。  

導入フレーズを追加しないでください。応答は応答から始まり、見出し: ### 拡張された ...
''',
    llm_config=gpt4o_config_graph,
    description='科学者によって作成された研究提案の「設計原則」側面を拡張できます。',
)

unexpected_properties_agent = AssistantAgent(
    name="unexpected_properties_agent",
    system_message = '''unexpected_properties_agent. 研究提案のこの特定の側面: ```{予期しない特性}```を慎重に拡張します。

元の内容を批判的に評価し、改善します。 \
より具体的な定量的な科学情報（化学式、数値、シーケンス、処理条件、微細構造など）を追加し、 \
根拠と段階的な推論を追加します。可能な場合は、特定のモデリングおよびシミュレーション技術、実験方法、または特定の分析についてコメントします。

まず、次の内容の科学を批判的に評価し、改善することを任務とするピアレビュアーの視点から、この初期ドラフトを慎重に評価します:

<予期しない特性>
ここで<予期しない特性>は研究提案の予期しない特性の側面です。  

導入フレーズを追加しないでください。応答は応答から始まり、見出し: ### 拡張された ...
''',
    llm_config=gpt4o_config_graph,
    description='科学者によって作成された研究提案の「予期しない特性」側面を拡張できます。',
)

comparison_agent = AssistantAgent(
    name="comparison_agent",
    system_message = '''comparison_agent. 研究提案のこの特定の側面: ```{比較}```を慎重に拡張します。

元の内容を批判的に評価し、改善します。 \
より具体的な定量的な科学情報（化学式、数値、シーケンス、処理条件、微細構造など）を追加し、 \
根拠と段階的な推論を追加します。可能な場合は、特定のモデリングおよびシミュレーション技術、実験方法、または特定の分析についてコメントします。

まず、次の内容の科学を批判的に評価し、改善することを任務とするピアレビュアーの視点から、この初期ドラフトを慎重に評価します:

<比較>
ここで<比較>は研究提案の比較の側面です。  

導入フレーズを追加しないでください。応答は応答から始まり、見出し: ### 拡張された ...
''',
    llm_config=gpt4o_config_graph,
    description='科学者によって作成された研究提案の「比較」側面を拡張できます。',
)

novelty_agent = AssistantAgent(
    name="novelty_agent",
    system_message = '''novelty_agent. 研究提案のこの特定の側面: ```{新規性}```を慎重に拡張します。

元の内容を批判的に評価し、改善します。 \
より具体的な定量的な科学情報（化学式、数値、シーケンス、処理条件、微細構造など）を追加し、 \
根拠と段階的な推論を追加します。可能な場合は、特定のモデリングおよびシミュレーション技術、実験方法、または特定の分析についてコメントします。

まず、次の内容の科学を批判的に評価し、改善することを任務とするピアレビュアーの視点から、この初期ドラフトを慎重に評価します:

<新規性>
ここで<新規性>は研究提案の新規性の側面です。  

導入フレーズを追加しないでください。応答は応答から始まり、見出し: ### 拡張された ...
''',
    llm_config=gpt4o_config_graph,
    description='科学者によって作成された研究提案の「新規性」側面を拡張できます。',
)

critic_agent = AssistantAgent(
    name="critic_agent",
    system_message = '''critic_agent. あなたは正確で詳細かつ価値のある応答を提供する役に立つAIエージェントです。 

あなたは提案全体をすべての詳細と拡張された側面を読んで、次のことを提供します:

(1) ドキュメントの要約（1段落で、メカニズム、関連技術、モデルと実験、使用する方法などの詳細を含む）、

(2) 強みと弱み、改善の提案を含む徹底的な科学的レビュー。論理的な推論と科学的アプローチを含めます。

次に、このドキュメント内から、

(1) 分子モデリングで取り組むことができる最も影響力のある科学的質問を特定します。 \
\n\nそのようなモデリングとシミュレーションを設定して実行するための重要なステップを概説し、詳細を含め、計画された作業のユニークな側面を含めます。

(2) 合成生物学で取り組むことができる最も影響力のある科学的質問を特定します。 \
\n\nそのような実験作業を設定して実行するための重要なステップを概説し、詳細を含め、計画された作業のユニークな側面を含めます。

重要な注意:
***新規性と実現可能性を評価しないでください。新規性と実現可能性を評価しないでください。***
''',
    llm_config=gpt4o_config_graph,
    description='7つの側面すべてがエージェントによって拡張された後、要約、批評、および改善の提案を行うことができます。',
)


novelty_assistant = autogen.AssistantAgent(
    name="novelty_assistant",
    system_message = '''あなたは研究提案の潜在的な影響を評価するために科学者のグループと協力する重要なAIアシスタントです。あなたの主なタスクは、提案された研究仮説の新規性と実現可能性を評価し、既存の文献と大きく重複しないこと、またはすでに十分に探求されている領域に踏み込まないことを確認することです。

Semantic Scholar APIにアクセスして関連文献を調査し、任意の検索クエリの上位10件の結果とその要約を取得できます。この情報に基づいて、アイデアを厳密に評価し、新規性と実現可能性を1から10のスケールで評価します（1が最低、10が最高）。

特に新規性に関しては厳格な評価者であることを目指してください。新しい会議や査読付き研究論文を正当化できる十分な貢献があるアイデアのみがあなたの審査を通過するべきです。

慎重に分析した後、新規性と実現可能性の評価を返します。

ツールの呼び出しが成功しなかった場合は、有効な応答が得られるまでツールを再呼び出してください。

評価後、推奨事項をまとめ、会話を終了するために "TERMINATE" と述べてください。''',
    llm_config=gpt4turbo_config,
)

# create a UserProxyAgent instance named "user_proxy"
novelty_admin = autogen.UserProxyAgent(
    name="novelty_admin",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    llm_config=False,
)

@novelty_admin.register_for_execution()
@novelty_assistant.register_for_llm(description='''この関数は、指定されたクエリに基づいてSemantic Scholar APIを使用して学術論文を検索するために設計されています。 
クエリは、関連するキーワードを+で区切って構成する必要があります。 ''')
def response_to_query(query: Annotated[str, '''論文検索のクエリ。クエリは関連するキーワードを+で区切って構成する必要があります。'''])->str:
    # APIエンドポイントURLを定義
    url = 'https://api.semanticscholar.org/graph/v1/paper/search'
    
    # より具体的なクエリパラメータ
    query_params = {
        'query': {query},           
        'fields': 'title,abstract,openAccessPdf,url'
                   }
    
    # APIキーを直接定義（リマインダー: 本番環境ではAPIキーを安全に取り扱ってください）
     # 実際のAPIキーに置き換えます
    
    # APIキーを含むヘッダーを定義
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    headers = {'x-api-key': api_key}
    
    # APIリクエストを送信
    response = requests.get(url, params=query_params, headers=headers)
    
    # レスポンスステータスを確認
    if response.status_code == 200:
       response_data = response.json()
       # 必要に応じてレスポンスデータを処理して表示
    else:
       response_data = f"リクエストがステータスコード {response.status_code} で失敗しました: {response.text}"

    return response_data

@user.register_for_execution()
@planner.register_for_llm()
@assistant.register_for_llm(description='''この関数は知識パスを作成するために使用できます。関数は入力として2つのキーワードを取るか、ランダムに割り当てることができ、次にこれらのノード間のパスを返します。 
パスにはいくつかの概念（ノード）とそれらの間の関係（エッジ）が含まれます。関数はパスを返します。
パスがすでに提供されている場合は、この関数を使用しないでください。パスもキーワードも提供されていない場合は、キーワードをNoneに選択して、ランダムに選択されたノード間のパスを生成します。''')
def generate_path(keyword_1: Annotated[Union[str, None], '知識グラフの最初のノード。ランダム選択の場合はNone。'],
                    keyword_2: Annotated[Union[str, None], '知識グラフの2番目のノード。ランダム選択の場合はNone。'],
                 ) -> str:
    
    path_list_for_vis, path_list_for_vis_string = create_path(G, embedding_tokenizer,
                                    embedding_model, node_embeddings , generate_graph_expansion=None,
                                    randomness_factor=0.2, num_random_waypoints=4, shortest_path=False,
                                    second_hop=False, data_dir='./', save_files=False, verbatim=True,
                                    keyword_1 = keyword_1, keyword_2=keyword_2,)

    return path_list_for_vis_string

@user.register_for_execution()
@planner.register_for_llm()
@assistant.register_for_llm(description='''この関数を使用して、研究アイデアの新規性と実現可能性を文献と比較して評価します。関数はSemantic Scholarを使用して文献記事にアクセスします。  
関数は1から10の範囲で新規性と実現可能性の評価を返します（最低から最高）。関数の入力は詳細な仮説です。''')
def rate_novelty_feasibility(hypothesis: Annotated[str, '研究仮説。']) -> str:
    res = novelty_admin.initiate_chat(
    novelty_assistant,
        clear_history=True,
        silent=False,
        max_turns=10,
    message=f'''次の研究仮説を評価してください\n\n{hypothesis}. \n\n関数を最大3回呼び出しますが、並行して呼び出さないでください。結果を待ってから次の関数を呼び出してください。 ''',
        summary_method="reflection_with_llm",
        summary_args={"summary_prompt" : "分析結果をそのまま返します。"}
    )

    return res.summary


planner.reset()
assistant.reset()
ontologist.reset()
scientist.reset()
critic_agent.reset()


groupchat = autogen.GroupChat(
    agents=[user, planner, assistant, ontologist, scientist,
            hypothesis_agent, outcome_agent, mechanism_agent, design_principles_agent, unexpected_properties_agent, comparison_agent, novelty_agent, critic_agent#sequence_retriever,
               ], messages=[], max_round=50, admin_name='user', send_introductions=True, allow_repeat_speaker=True,
    speaker_selection_method='auto',
)

manager = autogen.GroupChatManager(groupchat=groupchat, 
                                   llm_config=gpt4turbo_config, 
                                   system_message='あなたは動的に話者を選択します。')