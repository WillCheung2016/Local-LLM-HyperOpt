
import json
import re
import time

from ollama_client import OllamaChatClient

class LLMOptimizer:
    def __init__(
        self,
        is_expert="generic",
        model="llama3",
        temperature=0.0,
        max_tokens=600,
        frequency_penalty=0.0,
        use_cot=False,
        base_url=None,
    ):
        self.model = model
        self.is_expert = is_expert
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.use_cot = use_cot
        self.messages = []
        self.client = OllamaChatClient(model=model, base_url=base_url)

        # Initialize the conversation with the given system prompt
        self.initial_config()

    def initial_config(self):
        """This expert prompt does not seem to make too much of a difference (see Appendix in paper), but conditioning on good performance is generally a good idea."""
        if self.is_expert == "generic":
            message = {"role":"system", "content": "You are a machine learning expert."}
            self.messages.append(message)

    def call_llm(self, max_retries=2):
        tries = 0
        while tries < max_retries:
            try:
                response_text = self.client.chat(
                    self.messages,
                    temperature=self.temperature,
                    num_predict=self.max_tokens,
                    frequency_penalty=self.frequency_penalty,
                )
                self.messages.append({"role": "assistant", "content": response_text})
                return response_text
            except Exception as e:
                tries += 1
                print(e)
                time.sleep(5)
        raise Exception("Failed to call LLM, max retries exceeded")

    def _parse_raw_message(self, raw_message):
        # Parse the raw message into model source code, optimizer source code, and hparams
        json_match = re.search(r'```json\n(.*)\n```', raw_message, re.DOTALL)
        if json_match is None:
            raise Exception("Failed to parse raw message")
        params = json.loads(json_match.group(1).strip())
        assert isinstance(params, dict)
        assert "x1" in params and "x2" in params
        
        return params
    
    def parse_message(self, raw_message):
        if "Output: " in raw_message:
            raw_message = raw_message.split("Output: ")[1]
        cleaned_message = raw_message.strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_message, re.DOTALL)
        if fenced_match:
            cleaned_message = fenced_match.group(1)
        try:
            params = json.loads(cleaned_message)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", cleaned_message, re.DOTALL)
            if json_match:
                params = json.loads(json_match.group(0))
            else:
                print("***Raising exception...")
                print(raw_message)
                raise Exception("Failed to parse message")
        params = params["x"]
        return params
        
    
    def ask(self, prompt):
        self.messages.append({"role":"user", "content": prompt})
        raw_message = self.call_llm()
        params = self.parse_message(raw_message)
        return params
    
    def get_current_messages(self):
        return self.messages