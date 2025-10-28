from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import os
import dotenv
import re
import json
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
        "reason": "<Your analysis for the ideology detection, including which dimensions are related, which are unrelated, and how the text expressed the stance in these related dimensions.>",
        "result":{{
            "I1": <The label number>,
            "I2": <The label number>,
            "I3": <The label number>,
            "I4": <The label number>,
            "I5": <The label number>,
            "I6": <The label number>,
            "I7": <The label number>,
            "I8": <The label number>,
            "I9": <The label number>,
            "I10": <The label number>,
            "I11": <The label number>,
            "I12": <The label number>
        }}
    }}
    **Meaning of Each Label Number:**
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
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=text)
                ],
            )

            return response.content
        except Exception as e:
            print(e)
            print(response)

    
    async def ainvoke(self, text):
        try:
            response = await self.llm.ainvoke(
                 [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=text)
                ],
            )

            return response.content
        
        except Exception as e:
            print(e)


def parse_json(text):
    text = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.DOTALL)
    return json.loads(text)