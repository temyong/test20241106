import pandas as pd
import re
import openai
import time
import datetime
import os
import sys
import vertexai
import json
from google.cloud.exceptions import GoogleCloudError
from vertexai.preview.generative_models import GenerativeModel
import yaml

config = yaml.safe_load(open("config.yaml"))

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ModelAdapter:
    def __init__(self, vendor, max_retries=5):
        self.max_retries = max_retries
        self.vendor = vendor

        if vendor == "gemini":
            MODEL_ID = config["gemini"]["model_id"]
            PROJECT_ID = config["gemini"]["project_id"]
            REGION = config["gemini"]["region"]

            print(f"{bcolors.WARNING}calling gemini {MODEL_ID}{bcolors.ENDC}")
            vertexai.init(project=PROJECT_ID, location=REGION)
            self.gemini_model = GenerativeModel(
                model_name=MODEL_ID,
                system_instruction=[
                    "You are an assistant that helps verify word usage."
                ],
            )
        else:
            openai.api_key = config["openai"]["api_key"]
            print(f"{bcolors.WARNING}calling openai{bcolors.ENDC}")

    def invoke_prompt(self, prompt):
        if self.vendor == "gemini":
            return self.call_gemini_api(prompt)
        else:
            return self.call_openai_api(prompt)

    def call_gemini_api(self, prompt):
        for attempt in range(self.max_retries):
            try:
                response = self.gemini_model.generate_content([prompt])
                reply = response.json_object
                print(reply)
                return reply
            except GoogleCloudError as e:
                print(f"{bcolors.ERROR}Error: {e}. Retrying in 10 seconds... (Attempt {attempt + 1} of {self.max_retries}){bcolors.ENDC}")
                time.sleep(10)
        raise Exception(f"Failed to call Google Gemini after {self.max_retries} attempts")

    def call_openai_api(self, prompt):
        for attempt in range(self.max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-2024-08-06",
                    messages=[
                        {"role": "system", "content": "You are an assistant that helps verify word usage."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={ "type": "json_object" }
                )
                reply = json.loads(response['choices'][0]['message']['content'])
                print(reply)
                return reply
            except openai.error.OpenAIError as e:
                print(f"{bcolors.ERROR}Error: {e}. Retrying in 10 seconds... (Attempt {attempt + 1} of {self.max_retries}){bcolors.ENDC}")
                time.sleep(10)
        raise Exception(f"Failed to call OpenAI API after {self.max_retries} attempts")
        
        
def process_csv(vendor, input_csv, output_csv):
    df = pd.read_csv(input_csv, encoding="utf-8")
    model_adapter = ModelAdapter(vendor)

    df['reply'] = ""

    for index, row in df.iterrows():
        try:
            print(f"{bcolors.OKBLUE}INFO: Processing row {index}{bcolors.ENDC}")
            prompt = f"""
            list all meaning senses for the word '{row['label']}' as it would be used in an English learners dictionary.
            For each meaning sense provide:
            1. the baseform of the word (If the most common form of a particular meaning sense is a multiword or phrasal verb, etc show the Baseform in its multiword form). label this as baseForm
            2. a simple learners definition. label this as longDefinition
            3. A short form of that learners definition. label this as shortDefinition
            4. a simple sample sentence that provides good context using the word in its baseform, using using words no more complex than necesssary; and avoiding proper nouns). label this as exampleSentence
            5. a short, de-contextualized sample sentence, using the word in its baseform. label this as exampleSentenceShort
            6. part of speech (as it appears in the exampleSentenceShort)
            7. guideword(s). keep it as a comma-separated list instead of an array. label this as guideWords
            8. list both the CEFR Level as per Pearson Toolkit  and the Cambridge CEFR list. label this as cefrLevel
            9. a single word translation of the Baseform as it appears in the simple sample sentence in Spanish, Japanese, Korean, Portuguese, Russian, Turkish, Vietnamese, Chinese, Taiwanese, Hebrew, Arabic, French, German, Polish. label this as $[language]Translation
            10. A relative frequency (in 0 to 100% terms) that each meaning sense represents of all the instances of the word form  your internal language model corpus. label this as relativeFrequency
            11. An absolute frequency which desribed in % terms the meaning sense appears versus all other words in your internal language model corpus. label this as absoluteFrequency

            Group these meaning senses in groups determined by grouping meaning sense that share enough meaning sense that a learned can likely guess one meaning sense from the other. label this grouping as 'senses'.
            apply this grouping even if there is only 1 group
            Also, please indicate which of meaning sense you list is the most common occuring in American English. give it an attribute called mostCommon which is a boolean value 1 or 0

            return the result in json format. use camel case for the json attributes. top level attribute should be meaningSenses
            """

            reply = model_adapter.invoke_prompt(prompt)
            df.at[index, 'reply'] = reply
            df.iloc[:index+1].to_csv(output_csv, index=False)
        except Exception as e:
            df.at[index, 'Error'] = e
            df.iloc[:index+1].to_csv(output_csv, index=False)

    df.to_csv(output_csv, index=False)
    print(f"{bcolors.OKGREEN}Processed CSV saved to {output_csv}{bcolors.ENDC}")


if __name__ == "__main__":
    # Set your OpenAI API key and file paths here
    start_time = time.perf_counter()

    input_csv = 'word_input.csv'  # Path to your input CSV file
    output_csv = 'word_output.csv'  # Path where the updated CSV will be saved
    try:
        vendor = sys.argv[1] or "openai"
    except:
        vendor = os.getenv("MODEL_VENDOR", "openai")

    process_csv(vendor, input_csv, output_csv)

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"{bcolors.OKGREEN}script finished in {datetime.timedelta(seconds=elapsed)}{bcolors.ENDC}")


