import random
from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format
import time
import os
import re
from anthropic import Anthropic
from openai import OpenAI
import boto3
import json
from typing import Dict, Optional
import time
import csv

SYSTEM_PROMPT = "You are a world class game player. We are playing a game of Diplomacy, a player brief will be provided with the prompt. Respond to your current situation as best you can with the intention of winning the game."
#scratch pad is a better summary than the summary so disabling this
SUMMARY_PROMPT = "Summarize this conversation, keep the phase and season information at the top."
SUMMARY_NAME = "claude-3-5-sonnet-latest" ##TODO Make this matter, rn hard coded to Austria
PHASES = ["CONVERSATION PHASE","RETREAT","BUILD"]

def read_prompt_template():
    with open("prompt_outline.txt", "r") as f:
        return f.read()

def load_models_config():
    models = {}
    with open("models-full-run.txt", "r") as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)  # Skip header
        for row in reader:
            model_provider, model_name, country, scratch_path, chat_path, all_output_path = [x.strip() for x in row]
            models[country] = {
                "model_provider" : model_provider,
                "model_name": model_name,
                "scratch_path": scratch_path,
                "chat_path": chat_path,
                "all_output_path": all_output_path
            }
    return models

class ModelStateManager:
    def __init__(self):
        self.models = load_models_config()
        self.prompt_template = read_prompt_template()
        os.makedirs('debug', exist_ok=True)
        self.api_clients = self._initialize_api_clients()

    def _initialize_api_clients(self) -> Dict[str, any]:
        api_keys = {
            'anthropic': os.getenv('ANTHROPIC_API_KEY'),
            'open_router': os.getenv('OPENROUTER_API_KEY'),
            'open_ai': os.getenv('OPENAI_4O_API_KEY'),
        }
        
        clients = {}
        try:
            if api_keys['anthropic']:
                clients['anthropic'] = Anthropic(api_key=api_keys['anthropic'])
                
            if api_keys['open_router']:
                clients['open_router'] = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_keys['open_router']
                )
                
            if api_keys['open_ai']:
                clients['open_ai'] = OpenAI(api_key=api_keys['open_ai'])
        except Exception as e:
            print(f"Error initializing API clients: {e}")
        return clients

    def get_scratchpad(self, country):
        try:
            with open(self.models[country]['scratch_path'], 'r') as f:
                return f.read()
        except FileNotFoundError:
            return ""
            
    def update_scratchpad(self, country, content):
        with open(self.models[country]['scratch_path'], 'a') as f:
            f.write(f"\n{content}\n")
            
    def generate_prompt(self, game, country, phase, phase_index = "", conv_file = None):
        full_game_state = self.get_game_state_text(game, country)
        current_phase = game.get_current_phase()
        
        prompt = self.prompt_template
        prompt = prompt.replace("[ACTIVE PHASE]", phase + phase_index)
        prompt = prompt.replace("[CURRENT GAME STATE]", full_game_state)

        scratch_content = self.get_scratchpad(country)
        scratch_section = scratch_content if scratch_content else 'No notes yet'
        prompt = prompt.replace("[SCRATCHPAD]", scratch_section)
        
        prompt = prompt.replace("[COUNTRY IDENTITY]", country)
        
        current_conversations = get_current_conversations(country)
        prompt = prompt.replace("[CURRENT CONVERSATIONS]", current_conversations)
        
        moves_header = """LIST OF POSSIBLE MOVES:
        - F = Fleet, A = Army
        - Syntax: [Unit Type][Location] [Order]
        - Orders: 
        - '-' = Move to 
        - 'S' = Support 
        - 'C' = Convoy
        - 'H' = Hold (default if no order)
        
        Available Orders:"""
        
        moves = self.get_available_moves(game, country)
        available_moves = "\n".join([f"- {move}" for move in moves])
        prompt = prompt.replace("[YOUR MOVES]", 
                            f"{moves_header}\n{available_moves}")
        
        with open('debug/current-prompt.txt', 'w') as f:
            f.write(prompt)
        
        return prompt

    def get_model_response(self, country: str, prompt: str) -> str:
        model_info = self.models[country]
        model_provider = model_info['model_provider']
        model_name = model_info['model_name']
        response_text = ""

        try:
            if model_provider == 'anthropic':
                response_text = self._call_anthropic(prompt, model_name)
            elif model_provider == 'open_router':
                response_text = self._call_open_router(prompt, model_name)
            elif model_provider == 'open_ai':
                response_text = self._call_openai(prompt, model_name)
            else:
                response_text = f"Unknown model {model_name} for {country}"
        except Exception as e:
            response_text = f"API Error: {str(e)}"
        
        self._save_full_response(country, response_text)
        cleaned_response = self._process_response(country, response_text)
        return cleaned_response

    def _process_response(self, country: str, response: str) -> str:
        scratch_content = self._extract_scratch_content(response)
        if scratch_content:
            with open(self.models[country]['scratch_path'], 'a') as f:
                f.write(f"\n{scratch_content}\n")
        cleaned_response = self._clean_response(response)
        print(f"\n=== {country} Writes ===" + "\n" + cleaned_response)
        return cleaned_response

    def _clean_response(self, response: str) -> str:
        cleaned = re.sub(r'<scratch>.*?</scratch>', '', response, flags=re.DOTALL)
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
        return cleaned.strip()

    def _extract_scratch_content(self, response: str) -> Optional[str]:
        match = re.search(r'<scratch>(.*?)</scratch>', response, re.DOTALL)
        return match.group(1).strip() if match else None

    def _save_full_response(self, country: str, response: str) -> None:

        output_path = self.models[country]['all_output_path']
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(f"\n=== {time.ctime()} ===\n{response}\n")

    def _call_anthropic(self, prompt: str, model_name: str) -> str:
        client = self.api_clients.get('anthropic')
        if not client:
            raise ValueError("Anthropic client not initialized")
        message = client.messages.create(
            model=model_name,
            max_tokens=2000,
            temperature= 0.1,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    def _call_open_router(self, prompt: str, model_name: str, max_retries: int = 100) -> str:
        client = self.api_clients.get('open_router')
        if not client:
            raise ValueError("OpenRouter client not initialized")
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                if hasattr(response.choices[0].message, 'content'):
                    return response.choices[0].message.content
                else:
                    print(f"Debug: Received null response on attempt {attempt + 1}, retrying in 2 seconds...")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return "Error: No content in response after all retries"
                    
            except Exception as e:
                print(f"OpenRouter API call failed on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    print("Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                return f"OpenRouter Error: {str(e)}"

    # def _call_open_router(self, prompt: str, model_name: str) -> str:
    #     client = self.api_clients.get('open_router')
    #     if not client:
    #         raise ValueError("OpenRouter client not initialized")
        
    #     try:
    #         response = client.chat.completions.create(
    #             model=model_name,
    #             messages=[
    #                 {"role": "system", "content": SYSTEM_PROMPT},
    #                 {"role": "user", "content": prompt}
    #             ],
    #             temperature=0.1,
    #             max_tokens=2000
    #         )
            
    #         if hasattr(response.choices[0].message, 'content'):
    #             return response.choices[0].message.content
    #         else:
    #             return "Error: No content in response"
                
    #     except Exception as e:
    #         print(f"OpenRouter API call failed: {str(e)}")
    #         return f"OpenRouter Error: {str(e)}"

    def _call_openai(self, prompt: str, model: str) -> str:
        client = self.api_clients.get('open_ai')
        if not client:
            raise ValueError("OpenAI client not initialized")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI Error: {str(e)}"
    
    def get_game_state_text(self, game, country):
        state = [
            f"=== CURRENT GAME STATUS ===",
            f"Season: {game.get_current_phase()}",
            f"Your Role: {country}",
            f"\nYOUR FORCES:",
            "\n".join(sorted(f"- {unit}" for unit in game.get_units(country))),
            f"\nCONTROLLED SUPPLY CENTERS:",
            ", ".join(sorted(game.get_centers(country))),
            f"\n\n=== FULL MAP OVERVIEW ===",
            "(SC=Supply Center, A=Army, F=Fleet)",
            "Format: Region (Type) [Owner] - Units/Status - Adjacent Regions",
            "-------------------------------------------------------------"
        ]
        for region in sorted(game.map.locs):
            region_info = []
            region_type = game.map.area_type(region).capitalize()
            if region in game.map.scs:
                region_type += " SC"
            owner = next((power for power, centers in game.get_centers().items() 
                          if region in centers), "Neutral")
            units = []
            for power, power_units in game.get_units().items():
                for unit in power_units:
                    if region in unit:
                        units.append(unit)
            adjacent = game.map.abut_list(region)
            region_info.append(f"{region} ({region_type})")
            region_info.append(f"[{owner}]")
            region_info.append("- Units: " + ", ".join(units) if units else "No units")
            region_info.append("Connects to: " + ", ".join(adjacent))
            state.append("\n".join(region_info) + "\n")
        return "\n".join(state)

    def get_phase_instructions(self, phase, phase_index): # not used currently
        instructions = {}
        return instructions.get(phase, "")
    
    def get_available_moves(self, game, country):
        orders = game.get_all_possible_orders()
        orderable = game.get_orderable_locations(country)
        moves = []
        for loc in orderable:
            if loc in orders:
                moves.extend(orders[loc])
        return moves

    def process_orders(self, game, country, response):
        availible_orders = self.get_available_moves(game, country)
        orders = [order for order in availible_orders if order in response]
        game.set_orders(country, orders)


def run_game():
    game = Game()
    state_manager = ModelStateManager()
    
    while not game.is_game_done:
        for phase in PHASES:
            if (phase == "CONVERSATION PHASE"):
                for i in [" 1 of 4", " 2 of 4", " 3 of 4", " 4 of 4"]: #, "2 of 3", "3 of 3" making it one round for testing
                    
                    for country in state_manager.models:
                        print(country, "CONVERSATION PHASE" + i)
                        prompt = state_manager.generate_prompt(game, country, phase, phase_index=i)
                        response = state_manager.get_model_response(country, prompt)
                        process_conversation(game, phase, i, country, response, state_manager)
                        if i == " 4 of 4":
                            state_manager.process_orders(game, country, response)

            elif game.get_current_phase().endswith('R') and phase == "RETREAT PHASE":
                for country in state_manager.models:
                    prompt = state_manager.generate_prompt(game, country, phase)
                    response = state_manager.get_model_response(country, prompt)
                    state_manager.process_orders(game, country, response)

            elif game.get_current_phase().endswith('B') and phase == "BUILD PHASE":
                for country in state_manager.models:
                    prompt = state_manager.generate_prompt(game, country, phase)
                    response = state_manager.get_model_response(country, prompt)
                    state_manager.process_orders(game, country, response)

           
            game.process()           
        to_saved_game_format(game, output_path='game.json', output_mode='w')
            


def process_conversation(game, phase, phase_index, sending_country, response, state_manager):
    countries = ["Austria", "England", "France", "Germany", "Italy", "Russia", "Turkey"]
    
    # 1) Get the block between "**Diplomatic Messages:**" and "**Orders:**"
    match = re.search(r"\*\*Diplomatic Messages:\*\*(.*?)\*\*Orders:\*\*", response, re.DOTALL)
    if not match:
        # If we can't find a Diplomatic Messages section, just return
        return
    
    diplo_block = match.group(1)
    messages = {}
    
    # 2) For each country, grab the text from "**To {country}:**" up to
    #    the next "**To " or "**Orders:**" or end of block.
    for country in countries:
        pattern = rf"\*\*To {country}:\*\*(.*?)(?=\*\*To|\*\*Orders|\Z)"
        found = re.search(pattern, diplo_block, re.DOTALL)
        if found:
            text = found.group(1).strip()
            # 3) If the message is exactly "[NO MESSAGE]" or empty, skip writing
            if text == "[NO MESSAGE]":
                text = ""
            if text:
                messages[country] = text
    
    # 4) Write each country's message to its corresponding file
    for country, msg in messages.items():
        # Skip if same as sender or no actual text
        if country == sending_country or not msg:
            continue
        
        # Construct file name: alphabetical order for consistent naming
        sorted_countries = sorted([country, sending_country])
        file_name = f"model-outputs/Messages_{sorted_countries[0]}_{sorted_countries[1]}.txt"
        
        # Append the new line
        phase_label = f"{phase}{phase_index}"
        with open(file_name, 'a') as f:
            f.write(f"{game.get_current_phase()} {phase_label} : {sending_country} - MESSAGE: {msg}\n")



# def process_conversation(game, phase, phase_index, sending_country, response, state_manager):


#     countries = ["Austria", "England", "France", "Germany", "Italy", "Russia", "Turkey"]
#     out = ""
#     for temp in countries :
#         if temp != sending_country: 
#             vals = [temp, sending_country]
#             vals.sort()
#             c1, c2 = vals[0], vals[1]
#             file = f"model-outputs/Messages_{c1}_{c2}.txt"
#             out = out + f"**Current conversation with {temp}:** \n"
#             with open(file, 'a') as f:
#                 f.write(f"{game.get_current_phase()} {phase + phase_index } : {sending_country} - MESSAGE: {message}\n")


#     #Just doing it by hand lol. TODO make number of responses variable
#     prompt = state_manager.generate_prompt(game, reciving_country, "REPLY", phase_index="1 of 5", conv_file= conv_file)
#     response = state_manager.get_model_response(reciving_country, prompt)
#     conversation_is_over = '<END>' in response
#     print("process 1")

#     with open(conv_file, 'a') as f:
#         f.write(f"{reciving_country} - REPLY 1 of 5: {response}\n")
    
#     if conversation_is_over: return
    
#     prompt = state_manager.generate_prompt(game, sending_country, "REPLY", phase_index="2 of 5", conv_file= conv_file)
#     response = state_manager.get_model_response(sending_country, prompt)
#     conversation_is_over = '<END>' in response

   
#     with open(conv_file, 'a') as f:
#         f.write(f"{sending_country} - REPLY 2 of 5: {message}\n")
#     print("process 2")
    
#     if conversation_is_over: return

#     prompt = state_manager.generate_prompt(game, reciving_country, "REPLY", phase_index="3 of 5", conv_file= conv_file)
#     response = state_manager.get_model_response(reciving_country, prompt)
#     conversation_is_over = '<END>' in response

#     with open(conv_file, 'a') as f:
#         f.write(f"{reciving_country} - REPLY 3 of 5: {response}\n")
#     print("process 3")

#     if conversation_is_over: return
    
#     prompt = state_manager.generate_prompt(game, sending_country, "REPLY", phase_index="4 of 5", conv_file= conv_file)
#     response = state_manager.get_model_response(sending_country, prompt)
#     conversation_is_over = '<END>' in response
#     with open(conv_file, 'a') as f:
#         f.write(f"{sending_country} - REPLY 4 of 5: {message}\n")
    
#     if conversation_is_over: return

#     prompt = state_manager.generate_prompt(game, reciving_country, "REPLY", phase_index="5 of 5", conv_file= conv_file)
#     response = state_manager.get_model_response(reciving_country, prompt)
#     conversation_is_over = '<END>' in response

#     with open(conv_file, 'a') as f:
#         f.write(f"{reciving_country} - REPLY 5 of 5: {response}\n")
    
#     return

def summarize_conversation(sending_country, reciving_country, state_manager):
    conv_file = f"conversation_{sending_country}_{reciving_country}.txt"
    with open(conv_file, 'r') as f:
                    current_conv = f.read()
    response = state_manager.get_model_response("Austria", SUMMARY_PROMPT + "\n" + current_conv )
    sending_chat_path = state_manager.models[sending_country]["chat_path"]
    reciving_country_chat_path = state_manager.models[reciving_country]["chat_path"]
    with open(sending_chat_path, "a") as f:
        f.write(response)
    with open(reciving_country_chat_path, "a") as f:
        f.write(response)

def save_model_outputs(game, country, response, state_manager, phase):
    model_info = state_manager.models[country]
    output = f"""
=== {game.get_current_phase()} ===
Country: {country}
Phase: {phase}
Response:
{response}
"""
    with open(model_info['all_output_path'], 'a') as f:
        f.write(output)


def get_current_conversations(country: str) -> str:
    countries = ["Austria", "England", "France", "Germany", "Italy", "Russia", "Turkey"]
    out = ""
    for temp in countries:
        if temp != country:
            vals = [temp, country]
            vals.sort()
            c1, c2 = vals[0], vals[1]
            file = f"model-outputs/Messages_{c1}_{c2}.txt"
            out = out + f"**Current conversation with {temp}:** \n"
            try:
                with open(file, 'r') as f:
                    out = out + f.read() + '\n'
            except FileNotFoundError:
                out = out
    return out

if __name__ == '__main__':
    run_game()

if __name__ == '__main__':
    run_game()