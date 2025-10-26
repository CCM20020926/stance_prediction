from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import os
import dotenv
dotenv.load_dotenv()

class LLMAgentZeroShot:
    system_prompt_template = '''You are a helpful assistant for ideology detection. You need to output the ideological detection results revealed in the user's input text 
    according to the given detection dimensions.
    Here is the given detection dimensions:
    {detection_dim}
    **Note:**
    You should return a JSON-formatted output without any additional texts, comments or code-block markers(such as ```json)!
    The JSON format should looks liks:
    {{
        "reason": "<Your analysis for the ideology detection>",
        "result":{{
            "I1": <The result number>,
            "I2": <The result number>,
            "I3": <The result number>,
            "I4": <The result number>,
            "I5": <The result number>,
            "I6": <The result number>,
            "I7": <The result number>,
            "I8": <The result number>,
            "I9": <The result number>,
            "I10": <The result number>,
            "I11": <The result number>,
            "I12": <The result number>
        }}
    }}
    **Meaning of Each Result Number:**
    - 0: Unrelated 
    - 1: Left
    - 2: Center (Nor Left or Right, while related)
    - 3: Right
    '''

    def __init__(self, model) -> None:
        with open('detection_dim.md', 'r') as f:
            detection_dim = f.read()

        self.system_prompt = self.system_prompt_template.format(
            detection_dim=detection_dim
        )

        self.llm = ChatOpenAI(
            model=model,
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0
        )


    def invoke(self, text):
        response = self.llm.invoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=text)
            ]
        )

        return response.content


    