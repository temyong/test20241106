import pandas as pd
import re
import openai
import time
import datetime
import os
import sys
import vertexai
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

class DataChecker:
    def __init__(self, api_key, input_csv, output_csv, vendor, max_retries=5):
        self.api_key = api_key
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.max_retries = max_retries
        self.vendor = vendor
        openai.api_key = self.api_key

        if vendor == "gemini":
            MODEL_ID = "gemini-1.5-flash-001"
            print(f"{bcolors.WARNING}calling gemini {MODEL_ID}{bcolors.ENDC}")
            self.gemini_model = GenerativeModel(
                model_name=MODEL_ID,
                system_instruction=[
                    "You are an assistant that helps verify word usage."
                ],
            )
        else:
            print(f"{bcolors.WARNING}calling openai{bcolors.ENDC}")

    def check_label_usage(self, label, canonical_line):
        pattern = rf'\b{re.escape(label)}\b'
        match = re.search(pattern, canonical_line, re.IGNORECASE)
        if not match:
            print(f"{bcolors.WARNING}WARN: Label '{label}' not found in '{canonical_line}', regex pattern '{pattern}'{bcolors.ENDC}")
        return bool(match)

    def check_base_form(self, label):
        prompt = f"Is the word '{label}' a base form. If it is not, reply with 'no', and provide a reason. Or if it is a multi word, a proper noun or a phrase, reply with 'yes'.Otherwise, reply with 'yes'."
        print(prompt)
        return self.invoke_prompt(prompt)

    def check_definition_usage(self, label, definition, canonical_line):
        prompt = f"Is the definition '{definition}' of the word '{label}' in the sentence '{canonical_line}' generally a correct defintion of the word '{label}'. If it is obviously incorrect, reply with 'no', and provide a reason. Otherwise, reply with 'yes'. Don't be too strict."
        print(prompt)
        return self.invoke_prompt(prompt)

    def check_pos(self, label, pos, canonical_line):
        prompt = f"in the sentence '{canonical_line}', is the word '{label}' used as a '{pos}'? Reply with 'yes' or 'no'. If 'no', explain the reason and provide the correct POS for the word in the sentence. In cases  where '{label}' is a phrase, as long as the type of phrase is correct e.g., abverb for abverbial phrase, etc. please assume it is correct."
        print(prompt)
        return self.invoke_prompt(prompt)

    def check_translation(self, label, japanese_translation, canonical_line):
        prompt = f"Is any of the Japanese words listed here: '{japanese_translation} an appropriate Japanese translation for the word '{label}' as its meaning is determined from the context of the sentence: '{canonical_line}'? Don't be too strict. Always reply with 'yes' or 'no'. If 'no', explain the reason."
        print(prompt)
        return self.invoke_prompt(prompt)

    def check_pos_japanese(self, japanese_translation, pos):
        prompt = f"Are any of the Japanese words listed here: '{japanese_translation} not in the grammatical form of a: {pos}"
        print(prompt)
        return self.invoke_prompt(prompt)

    def get_common_definition(self, label):
        prompt = f"Provide the most common definition of the word '{label}'."
        print(prompt)
        success, response = self.invoke_prompt(prompt)
        if success:
            return response.strip(), ""
        else:
            return "", "Error fetching definition"

    def compare_definitions(self, definition1, definition2):
        prompt = f"Are the following definitions similar? Definition 1: '{definition1}' Definition 2: '{definition2}' Reply with 'yes' or 'no'. If 'no', explain why not."
        print(prompt)
        success, response = self.invoke_prompt(prompt)
        if success and "yes" in response.lower():
            return True, ""
        else:
            return False, response

    def invoke_prompt(self, prompt):
        if self.vendor == "gemini":
            return self.call_gemini_api(prompt)
        else:
            return self.call_openai_api(prompt)

    def call_gemini_api(self, prompt):
        MODEL_ID = config["gemini"]["model_id"]
        PROJECT_ID = config["gemini"]["project_id"]
        REGION = config["gemini"]["region"]
        vertexai.init(project=PROJECT_ID, location=REGION)

        for attempt in range(self.max_retries):
            try:
                response = self.gemini_model.generate_content([prompt])
                reply = response.text.strip()
                if "yes" in reply.lower() or prompt.startswith("Provide the most common definition"):
                    return True, reply
                else:
                    return False, reply
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
                    temperature=0.2
                )
                reply = response['choices'][0]['message']['content'].strip()
                if "yes" in reply.lower() or prompt.startswith("Provide the most common definition"):
                    return True, reply
                else:
                    return False, reply
            except openai.error.OpenAIError as e:
                print(f"{bcolors.ERROR}Error: {e}. Retrying in 10 seconds... (Attempt {attempt + 1} of {self.max_retries}){bcolors.ENDC}")
                time.sleep(10)
        raise Exception(f"Failed to call OpenAI API after {self.max_retries} attempts")

    def process_csv(self):
        df = pd.read_csv(self.input_csv, encoding="utf-8")

        df['LabelMismatch'] = 0
        df['IsBaseForm'] = 1
        df['BaseFormReason'] = 1
        df['DefMismatch'] = 0
        df['DefMismatchReason'] = ""
        df['POSMismatch'] = 0
        df['POSMismatchReason'] = ""
        df['TranslationMismatch'] = 0
        df['TranslationMismatchReason'] = ""
        df['CommonDefinition'] = ""
        df['CommonDefMismatch'] = 0
        df['CommonDefMismatchReason'] = ""
        df['Error'] = ""

        df = df[[
            'rank',
            'id',
            'label',
            'canonicalLine',
            'LabelMismatch',
            'definition',
            'IsBaseForm',
            'BaseFormReason',
            'CommonDefMismatch',
            'CommonDefinition',
            'CommonDefMismatchReason',
            'DefMismatch',
            'DefMismatchReason',
            'japaneseTranslation',
            'TranslationMismatchReason',
            'TranslationMismatch',
            'POS',
            'POSMismatch',
            'POSMismatchReason'
        ]]

        for index, row in df.iterrows():
            try:
                print(f"{bcolors.OKBLUE}INFO: Processing row {index}{bcolors.ENDC}")
                label = row['label']
                definition = row['definition']
                pos = row['POS']
                japanese_translation = row['japaneseTranslation']
                canonical_line = row['canonicalLine']

                reasons = []

                if not self.check_label_usage(label, canonical_line):
                    df.at[index, 'LabelMismatch'] = 1

                baseform_result, baseform_reason = self.check_base_form(label)
                print(f"{bcolors.OKCYAN} baseform_result={baseform_result}, baseform_reason={baseform_reason}{bcolors.ENDC}")
                if not baseform_result:
                    df.at[index, 'IsBaseForm'] = 0
                    df.at[index, 'BaseFormReason'] = baseform_reason

                definition_matched, definition_reason = self.check_definition_usage(label, definition, canonical_line)
                print(f"{bcolors.OKCYAN} definition_matched={definition_matched}, definition_reason={definition_reason}{bcolors.ENDC}")
                if not definition_matched:
                    df.at[index, 'DefMismatch'] = 1
                    df.at[index, 'DefMismatchReason'] = definition_reason

                pos_matched, pos_reason = self.check_pos(label, pos, canonical_line)
                print(f"{bcolors.OKCYAN} pos_matched={pos_matched}, pos_reason={pos_reason}{bcolors.ENDC}")
                if not pos_matched:
                    df.at[index, 'POSMismatch'] = 1
                    df.at[index, 'POSMismatchReason'] = pos_reason

                translation_matched, translation_reason = self.check_translation(label, japanese_translation, canonical_line)
                print(f"{bcolors.OKCYAN} translation_matched={translation_matched}, translation_reason={translation_reason}{bcolors.ENDC}")
                if not translation_matched:
                    df.at[index, 'TranslationMismatch'] = 1
                    df.at[index, 'TranslationMismatchReason'] = translation_reason

                pos_translation_matched, pos_translation_reason = self.check_pos_japanese(japanese_translation, pos)
                print(f"{bcolors.OKCYAN} pos_translation_matched={pos_translation_matched}, pos_translation_reason={pos_translation_reason}{bcolors.ENDC}")
                if not pos_translation_matched:
                    df.at[index, 'POSTranslationMismatch'] = 1
                    df.at[index, 'POSTranslationMismatchReason'] = pos_translation_reason

                # Get the most common definition from GPT
                #common_definition, error_msg = self.get_common_definition(label)
                #print(f"{bcolors.OKCYAN} common_definition={common_definition}, definition_error={error_msg}{bcolors.ENDC}")
                #if error_msg:
                #    df.at[index, 'CommonDefMismatchReason'] = error_msg
                #else:
                #    df.at[index, 'CommonDefinition'] = common_definition
                #    definitions_match, compare_reason = self.compare_definitions(common_definition, definition)
                #    print(f"{bcolors.OKCYAN} definitions_match={definitions_match}, compare_reason ={compare_reason}{bcolors.ENDC}")
                #    if not definitions_match:
                #        df.at[index, 'CommonDefMismatch'] = 1
                #        df.at[index, 'CommonDefMismatchReason'] = compare_reason

                # Save progress to CSV after processing each row
                df.iloc[:index+1].to_csv(self.output_csv, index=False)
            except Exception as e:
                df.at[index, 'Error'] = e
                df.iloc[:index+1].to_csv(self.output_csv, index=False)

        df.to_csv(self.output_csv, index=False)
        print(f"{bcolors.OKGREEN}Processed CSV saved to {self.output_csv}{bcolors.ENDC}")


if __name__ == "__main__":
    # Set your OpenAI API key and file paths here
    start_time = time.perf_counter()
    api_key = config["openai"]["api_key"]

    input_csv = 'ngsl_check_input.csv'  # Path to your input CSV file
    output_csv = 'ngsl_check_output.csv'  # Path where the updated CSV will be saved
    try:
        vendor = sys.argv[1] or "openai"
    except:
        vendor = os.getenv("MODEL_VENDOR", "openai")
    checker = DataChecker(api_key, input_csv, output_csv, vendor)
    checker.process_csv()

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"{bcolors.OKGREEN}script finished in {datetime.timedelta(seconds=elapsed)}{bcolors.ENDC}")


