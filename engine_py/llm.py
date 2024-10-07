from transformers import pipeline


class LLM:
    def __init__(self, model: str = "facebook/opt-125m", device: str = "cpu") -> None:
        """Class acting as a wrapper for the LLM and the response generation pipeline.

        Args:
            model (str): Model used to generate the responses.
            device (str, optional): Where to run the model (e.g.: cpu, cuda etc.). Defaults to "cpu".

        Raises:
            ValueError: Raises an error if the model is not supported.
        """
        if model == "facebook/opt-125m":
            self.model_name = model
            # self.tokenizer = AutoTokenizer.from_pretrained(model)
            # self.model = AutoModelForCausalLM.from_pretrained(model)
            self.pipeline = pipeline(model=self.model_name)
        else:
            raise ValueError(f"Model {model} not supported.")
        self.device = device
        # self.model.to(self.device)
        
    def tokenize(self, prompt: str) -> dict:
        """tokenize the prompt using self.tokenizer and transfer the tensors to self.device and then return the tokenized prompt.

        Args:
            prompt (str): Input prompt in string format

        Returns:
            dict: Tokenized prompt in dictionary format
        """
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def generate(self, prompt: str) -> str:
        """Generate text for the given prompts using the huggingface pipeline and the model given by self.model.

        Args:
            prompts str: Input prompt in string format

        Returns:
            str: Generated text in string format
        """
        response = self.pipeline(prompt)
        print(response)
        return response[0]['generated_text']
    
    def batch_process(self, prompts: list[str]) -> list[str]:
        """Process a batch of prompts using the generate method.

        Args:
            prompts (list[str]): List of prompts to process

        Returns:
            list[str]: List of generated texts
        """
        batch_response = []
        for out in self.pipeline(prompts, batch_size=len(prompts), max_length=1000):
            print(out)
            batch_response.append(out[0]['generated_text'])
        return batch_response
        
        