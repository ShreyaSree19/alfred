import sys
import os
import time
import threading
import random

import torch
from torch.nn import functional as F
import transformers
from accelerate import infer_auto_device_map, disk_offload
import numpy as np

from utils import load_video_frames_cv2, get_video_metadata

from typing import Dict, List, AnyStr, Union



# import cv2

"""
Interface for LLM
Contains the following:
- model_name: path to model
- cuda_devices
- tokenizer
- save_dir # IF YOU ARE RUNNING A 70B MODEL, YOU NEED TO CHANGE THIS TO A FOLDER IN YOUR PRIVATE SPACE (and make sure the folder exists)
"""


class transformers_interface:
	def __init__(
		self,
		model_name: str = "/media/atkeonlab-3/Mass Storage/models/Llama-3-8b-chat-hf",
		cuda_devices: List[int] = [0, 1],
		vision_model: bool = False,
		save_dir: str = "/media/atkeonlab-3/Mass Storage/models/cache",
	):
		self.vision_model = vision_model
		self.model_name = model_name
		self.cuda_devices = cuda_devices
		print("Loading model: ", model_name)
		self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

		

		self.save_dir = save_dir
		# try:
		#     os.mkdir(save_dir)
		# except FileExistsError:
		#     pass
		if vision_model:
			self.model = transformers.MllamaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
			self.processor = transformers.AutoProcessor.from_pretrained(model_name)
			self.eos_token_id = [128001, 128009]
		else:

			mem_map = {}
			for i in range(torch.cuda.device_count()):
				device_mem = torch.cuda.get_device_properties(i).total_memory
				if i in cuda_devices:
					mem_map[i] = "%iGiB" % (device_mem // 1024**3)
				else:
					mem_map[i] = "0GiB"
			self.model = transformers.AutoModelForCausalLM.from_pretrained(
				model_name,
				low_cpu_mem_usage=True,
				device_map="balanced",
				max_memory=mem_map,
				offload_folder=save_dir,
				torch_dtype=torch.bfloat16,
			)
			self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
		self.model.eval()
		
					

		self.yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
		self.no_token_id = self.tokenizer.convert_tokens_to_ids("no")
		self.letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		self.letter_token_ids = [self.tokenizer.convert_tokens_to_ids(letter) for letter in self.letters]

		

	def get_num_tokens(self, prompt):
		assert isinstance(prompt, str)
		batch = self.tokenizer(prompt, return_tensors="pt")
		batch = {k: v.cuda() for k, v in batch.items()}
		return batch["input_ids"].shape[1]

	def gen_inputs(self, prompt, image):
		inputs = self.processor(image, prompt, return_tensors="pt")
		inputs = {k: v.cuda() for k, v in inputs.items()}
		return inputs


	"""
	generates next set of tokens from prompt
	"""

	def generate(
		self,
		prompt: Union[str, List[str]],
		num_tokens=400,
		top_p=0.9,
		sampling=True,
		stopword=None,
	) -> Union[str, List[str]]:
		is_batch = isinstance(prompt, list)

		if is_batch:
			context = [str(e).rstrip().lstrip() for e in prompt]
			batch: Dict = self.tokenizer(context, return_tensors="pt", padding=True)
		else:
			context = str(prompt).rstrip().lstrip()
			batch: Dict = self.tokenizer(context, return_tensors="pt")

		batch = {k: v.cuda() for k, v in batch.items()}

		output = self.model.generate(
			**batch,
			do_sample=sampling,
			top_p=top_p,
			repetition_penalty=1.1,
			max_new_tokens=num_tokens,
			eos_token_id=self.tokenizer.eos_token_id,
		)

		# if the prompt is a list
		if is_batch:
			output_text = self.tokenizer.batch_decode(
				output[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
			)
		else:
			output_text = self.tokenizer.batch_decode(
				output[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
			)[0]
			if stopword is not None and stopword in output_text:
				output_text = output_text[: output_text.index(stopword)] + stopword
		return output_text

	def generate_chat(self, messages, num_tokens=400, sampling=True):

		batch = self.tokenizer.apply_chat_template(
			messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
		).cuda()
		# print("Batch device: ", batch.get_device())

		output = self.model.generate(
			batch,
			do_sample=sampling,
			max_new_tokens=num_tokens,
			eos_token_id=self.tokenizer.eos_token_id,
		)
		output_text = self.tokenizer.batch_decode(
			output[:, batch.shape[1] :], skip_special_tokens=True
		)
		return output_text

	def generate_chat_easy(self,sys_prompt,user_prompt,num_tokens=400,sampling=True):
		messages = [
			{
				"role": "system",
				"content": sys_prompt,
			},
			{
				"role": "user",
				"content": user_prompt,
			},
		]
		return self.generate_chat(messages,num_tokens=num_tokens,sampling=sampling)

	def generate_vision(self, prompt, image, max_new_tokens=80, stop_strings = None):
		assert self.vision_model
		inputs = self.gen_inputs(prompt, image)
		generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, eos_token_id=self.eos_token_id)
		return self.processor.batch_decode(
			generate_ids[:, inputs["input_ids"].shape[1] :],
			skip_special_tokens=True,
			clean_up_tokenization_spaces=False,
		)[0]

	def generate_choice(self, prompt, choices, sample=False):
		probs = self.eval_log_probs(prompt, choices)
		probs = probs / np.sum(probs)
		if not sample:
			return choices[np.argmax(probs)], probs
		else:
			probs = probs / np.sum(probs)
			return np.random.choice(choices, p=probs), probs

	"""
	Generate next set of tokens from input prompt until stopword is generated
	"""

	def generate_stopword_and_verify(
		self, prompt: str, stopword: str, max_attempts: int = 5
	):
		# loop until we find a stopword or we have reached the maximum number of attempts
		success = False
		while not success and max_attempts > 0:
			new_text = self.generate(
				prompt, num_tokens=64, use_cache=False, stopword=stopword
			)
			if stopword in new_text:
				success = True
			else:
				max_attempts -= 1
		return new_text, success

	"""
	return tokens & logprobs of tokens
	"""

	def to_tokens_and_logprobs(
		self, input_texts: List[str], return_tokens: bool = False
	):
		batch = self.tokenizer(input_texts, padding=True, return_tensors="pt")
		input_ids = batch["input_ids"]
		batch = {k: v.cuda() for k, v in batch.items()}
		outputs = self.model(**batch)
		probs = torch.log_softmax(outputs.logits, dim=-1).detach().cpu()

		# collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
		probs = probs[:, :-1, :]
		input_ids = input_ids[:, 1:]
		gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

		batch = []
		for input_sentence, input_probs in zip(input_ids, gen_probs):
			text_sequence = []
			for token, p in zip(input_sentence, input_probs):
				if token not in self.tokenizer.all_special_ids:
					if return_tokens:
						text_sequence.append((self.tokenizer.decode(token), p.item()))
					else:
						text_sequence.append(p.item())
			batch.append(text_sequence)
		return batch

	"""
	returns the logprobs of generating the specified queries from the given prompt
	"""

	def eval_log_probs(
		self,
		prompt: str,
		queries: List[str],
		normalize_by_length: bool = True,
		batch_size: int = None,
	):
		prompt_tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
		num_prompt_tokens = (
			np.sum(
				[
					(
						1
						if prompt_tokens[0, i] not in self.tokenizer.all_special_ids
						else 0
					)
					for i in range(prompt_tokens.shape[1])
				]
			)
			- 1
		)
		print("Num prompt tokens: {}".format(num_prompt_tokens))
		sequences = [prompt + query for query in queries]
		if batch_size is not None:
			log_probs = []
			for i in range(0, len(sequences), batch_size):
				log_probs += self.to_tokens_and_logprobs(sequences[i : i + batch_size])

		else:
			log_probs = self.to_tokens_and_logprobs(sequences)
		probs = np.zeros(len(queries))
		for i in range(len(queries)):
			prob = np.sum(log_probs[i][num_prompt_tokens:])
			if normalize_by_length:
				prob = prob / (len(log_probs[i]) - num_prompt_tokens)
			probs[i] = np.exp(prob)
		return probs

	def yes_no_question(self, prompt):
		inputs = self.tokenizer(prompt, return_tensors="pt")
		inputs = {k: v.cuda() for k, v in inputs.items()}
		outputs = self.model.generate(
			**inputs,
			max_new_tokens=1,
			do_sample=True,
			return_dict_in_generate=True,
			output_scores=True,
			output_logits=True,
			eos_token_id=self.tokenizer.eos_token_id,
		)
		logits = outputs["logits"][0]
		probs = F.softmax(logits, dim=-1)
		yes_prob = probs[0][self.yes_token_id].item()
		no_prob = probs[0][self.no_token_id].item()
		total_prob = yes_prob + no_prob
		# print("Yes prob: {}, No prob: {}".format(yes_prob, no_prob))
		yes_prob = yes_prob / total_prob
		no_prob = no_prob / total_prob
		return yes_prob, no_prob

	def yes_no_question_vision(self, image, prompt):
		assert self.vision_model
		inputs = self.gen_inputs(prompt, image)
		outputs = self.model.generate(
			**inputs,
			max_new_tokens=1,
			do_sample=True,
			return_dict_in_generate=True,
			output_scores=True,
			output_logits=True,
			# eos_token_id=self.tokenizer.eos_token_id,
		)
		logits = outputs["logits"][0]
		probs = F.softmax(logits, dim=-1)
		yes_prob = probs[0][self.yes_token_id].item()
		no_prob = probs[0][self.no_token_id].item()
		# print("Yes prob: {}, No prob: {}".format(np.exp(log_probs[0][self.yes_token_id].item()), np.exp(log_probs[0][self.no_token_id].item())))
		logits = outputs["logits"][0]
		log_probs = logits
		yes_prob = np.exp(log_probs[0][self.yes_token_id].item())
		no_prob = np.exp(log_probs[0][self.no_token_id].item())


		total_prob = yes_prob + no_prob
		# print("Yes prob: {}, No prob: {}".format(yes_prob, no_prob))
		yes_prob = yes_prob / total_prob
		no_prob = no_prob / total_prob
		return yes_prob, no_prob

	def yes_no_question_chat(self, prompt):
		messages = [
			{
				"role": "system",
				"content": "You are an assistant designed to help people with household tasks. Do not be verbose. Answer the question with no added qualifications or caveats. Just directly provide the answer.",
			},
			{
				"role": "user",
				"content": prompt,
			},
		]
		prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
		# print("Formatted text: ", formatted_text)
		inputs = self.tokenizer(prompt, return_tensors="pt")
		inputs = {k: v.cuda() for k, v in inputs.items()}
		outputs = self.model.generate(
			**inputs,
			max_new_tokens=1,
			do_sample=True,
			return_dict_in_generate=True,
			output_scores=True,
			output_logits=True,
			eos_token_id=self.tokenizer.eos_token_id,
		)
		logits = outputs["logits"][0]
		probs = F.softmax(logits, dim=-1)
		yes_prob = probs[0][self.yes_token_id].item()
		no_prob = probs[0][self.no_token_id].item()
		total_prob = yes_prob + no_prob
		# print("Yes prob: {}, No prob: {}".format(yes_prob, no_prob))
		yes_prob = yes_prob / total_prob
		no_prob = no_prob / total_prob
		return yes_prob, no_prob

	def mcq_question(self, prompt, choices):
		prompt += '\n'
		for i, choice in enumerate(choices):
			prompt += "{}. {}".format(self.letters[i], choice) + '\n'
		inputs = self.tokenizer(prompt, return_tensors="pt")
		inputs = {k: v.cuda() for k, v in inputs.items()}
		outputs = self.model.generate(
			**inputs,
			max_new_tokens=1,
			do_sample=True,
			return_dict_in_generate=True,
			output_scores=True,
			output_logits=True,
			eos_token_id=self.tokenizer.eos_token_id,
		)

		logits = outputs["logits"][0]
		probabilities = F.softmax(logits, dim=-1)
		probs = [probabilities[0][letter_id].item() for letter_id in self.letter_token_ids[:len(choices)]]
		probs = np.array(probs)
		probs = probs / np.sum(probs)
		return probs

	def mcq_question_chat(self, prompt, choices):
		prompt += '\n'
		for i, choice in enumerate(choices):
			prompt += "{}. {}".format(self.letters[i], choice) + '\n'

		# print("PROMPT BEFORE FORMATTING: ", prompt)

		messages = [
			{
				"role": "system",
				"content": "You are an assistant designed to help people with household tasks. Do not be verbose. Answer the question with no added qualifications or caveats. Just directly provide the answer.",
			},
			{
				"role": "user",
				"content": prompt,
			},
		]
		prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
		# print("PROMPT AFTER FORMATTING: ", prompt)
		# print(prompt)
		inputs = self.tokenizer(prompt, return_tensors="pt")
		inputs = {k: v.cuda() for k, v in inputs.items()}
		outputs = self.model.generate(
			**inputs,
			max_new_tokens=1,
			do_sample=True,
			return_dict_in_generate=True,
			output_scores=True,
			output_logits=True,
			eos_token_id=self.tokenizer.eos_token_id,
		)
		logits = outputs["logits"][0]
		probabilities = F.softmax(logits, dim=-1)
		probs = [probabilities[0][letter_id].item() for letter_id in self.letter_token_ids[:len(choices)]]
		probs = np.array(probs)
		probs = probs / np.sum(probs)
		return probs



	def action_probs_yn(self, prompt, actions, use_chat=False):
		if use_chat:
			probabilities = [self.yes_no_question_chat(prompt.replace("[action]", action))[0] for action in actions]
		else:
			probabilities = [self.yes_no_question(prompt.replace("[action]", action))[0] for action in actions]
		print(probabilities)
		probabilities = np.array(probabilities)
		probabilities = probabilities / np.sum(probabilities)
		return probabilities

	def action_probs_mcq(self, prompt, actions, shuffles = 0, use_chat=False):
		if shuffles > 1:
			action_dict = {}
			for action in actions:
				action_dict[action] = np.zeros(shuffles)
			shuffled_actions = actions.copy()
			for i in range(shuffles):
				np.random.shuffle(shuffled_actions)
				probabilities = self.mcq_question_chat(prompt, shuffled_actions) if use_chat else self.mcq_question(prompt, shuffled_actions)
				for j, action in enumerate(shuffled_actions):
					action_dict[action][i] = probabilities[j]
			probabilities = np.zeros(len(actions))
			for i in range(len(actions)):
				probabilities[i] = np.mean(action_dict[actions[i]])
		elif shuffles == 1:
			probabilities = self.mcq_question_chat(prompt, actions) if use_chat else self.mcq_question(prompt, actions)

		elif shuffles == 0:
			action_dict = {}
			for action in actions:
				action_dict[action] = np.zeros(len(actions))
			shuffled_actions = actions.copy()
			for i in range(len(actions)):
				new_shuffled_actions = shuffled_actions[1:]
				new_shuffled_actions.append(shuffled_actions[0])
				shuffled_actions = new_shuffled_actions
				probabilities = self.mcq_question_chat(prompt, shuffled_actions) if use_chat else self.mcq_question(prompt, shuffled_actions)
				for j, action in enumerate(shuffled_actions):
					action_dict[action][i] = probabilities[j]
			probabilities = np.zeros(len(actions))
			for i in range(len(actions)):
				probabilities[i] = np.mean(action_dict[actions[i]])

		probabilities = probabilities / np.sum(probabilities)
		
		return probabilities

	def SeViLa_VQA(self, query, frame_range, video_path, num_query_frames = 16, num_responses = 1):
		indices = np.linspace(frame_range[0], frame_range[1], num_query_frames).astype(int)
		clip = load_video_frames_cv2(video_path, indices)
		localizer_prompt = "<|image|><|begin_of_text|> Can the information in this image be used to accurately answer the question: " + query
		answer_confidences = [self.yes_no_question_vision(clip[i], localizer_prompt)[0] for i in range(num_query_frames)]
		# print([x for x in zip(indices, answer_confidences)])
		best_frame = indices[np.argmax(answer_confidences)]
		question_prompt = "<|image|><|begin_of_text|> " + query
		if num_responses == 1:
			answer = self.generate_vision(question_prompt, clip[np.argmax(answer_confidences)]).split(". ")[0] +"."
			return answer, best_frame
		else:
			answers = [self.generate_vision(question_prompt, clip[i]).split(". ")[0] + "." for i in range(num_responses)]
			return answers, best_frame
		

	def save_cache(self):
		pass



class Base_Server:
	def __init__(self, prompt_read_file, result_write_file, image_read_folder=None):
		self.prompt_read_file = prompt_read_file
		self.result_write_file = result_write_file
		self.image_read_folder = image_read_folder
		self.interface = None
		self.image_buffer = []

	def prompt_listener(self):
		print("Listening for prompt changes")

		with open(self.prompt_read_file, "r") as f:
			text = f.read()
		prev_uuid = text.split("\n")[0]

		while True:
			with open(self.prompt_read_file, "r") as f:
				text = f.read()
			uuid = text.split("\n")[0]
			prompt = "\n".join(text.split("\n")[1:])
			if uuid != prev_uuid:
				print("Prompt: {}".format(prompt))
				if prompt == "":
					print("recieved empty prompt, continuing")
					continue
				prev_uuid = uuid
				start_time = time.time()
				result = self.generate(prompt)
				print("Time taken: {}".format(time.time() - start_time))
				out = uuid + "\n" + result
				with open(self.result_write_file, "w") as f:
					f.write(out)
				print("Result: {}".format(result))
			time.sleep(0.25)

	def image_listener(self):
		# clear the image read folder
		for file in os.listdir(self.image_read_folder):
			if not file == "index.txt":
				os.remove(os.path.join(self.image_read_folder, file))
		self.image_buffer = []
		index_file = os.path.join(self.image_read_folder, "index.txt")
		with open(index_file, "r") as f:
			text = f.read()
		prev_uuid = text.split("\n")[0]
		print("Listening for new images")
		while True:
			with open(index_file, "r") as f:
				text = f.read()
			uuid = text.split("\n")[0]
			image_indicies = "\n".join(text.split("\n")[1:])
			if uuid != prev_uuid and image_indicies != "":
				prev_uuid = uuid
				image_indicies = [int(e) for e in image_indicies.split(",")]
				for image_index in image_indicies:
					image_file = os.path.join(
						self.image_read_folder, str(image_index).zfill(4) + ".jpg"
					)
					print("New image: {}".format(image_file))
					image = cv2.imread(image_file)  # type: ignore
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore
					if image is None:
						print("Image is none")
						continue
					else:
						print("Image shape: {}".format(image.shape))
					self.image_buffer.append(image)
			time.sleep(0.25)

	def generate(self, prompt):
		# raise NotImplementedError
		return "This is a test result"

	def run(self):
		self.prompt_listener()

	def run_image_listener(self):
		self.image_listener()


class LLM_Server(Base_Server):
	def __init__(self, prompt_read_file, result_write_file):
		super().__init__(prompt_read_file, result_write_file)
		self.interface = transformers_interface()

	def generate(self, prompt):
		return self.interface.generate(prompt, num_tokens=128)

def interactive_generate():
	LLAMA_PATH = "/home/mverghese/Models/Llama-3.2-11B-Vision-Instruct/"
	llm = transformers_interface(LLAMA_PATH, cuda_devices=[0, 1])
	prompt_file = "/home/mverghese/ThorCooking/test_prompt.txt"
	sys_prompt = "You are an assitant designed to help people with household tasks. Do not be verbose. Answer the question with no added qualifications or caveats. Just directly provide the answer."
	while True:
		input("Press enter to generate")
		with open(prompt_file, "r") as f:
			prompt = f.read()
		response = llm.generate_chat_easy(sys_prompt, prompt, num_tokens=512)
		print(response)



if __name__ == "__main__":
	# interactive_generate()
	LLAMA_PATH = "/home/mverghese/Models/Llama-3.2-11B-Vision-Instruct/"
	llm = transformers_interface(LLAMA_PATH, cuda_devices=[0, 1], vision_model=True)
	prompt_file = "/home/mverghese/ThorCooking/test_prompt.txt"
	with open(prompt_file, "r") as f:
		prompt = f.read()

	# messages = [
	# 	{
	# 		"role": "system",
	# 		"content": "You are an assitant designed to help people with household tasks. Do not be verbose. Answer the question with no added qualifications or caveats. Just directly provide the answer.",
	# 	},
	# 	{
	# 		"role": "user",
	# 		"content": prompt,
	# 	},
	# ]

	# prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

	actions = ["Pick up the Tomato", "Pick up the Lettuce", "Slice the Tomato", "Add the Tomato to the Sandwich", "Slice the Lettuce", "Add the Lettuce to the Sandwich"]
	probabilities = llm.action_probs_mcq(prompt, actions, use_chat=False)
	sorted_indices = np.argsort(probabilities)[::-1]
	print("Probabilites: ")
	for i in sorted_indices:
		print("{action}:\t{prob:10.3f}".format(action=actions[i], prob=probabilities[i]))

	probabilities = llm.action_probs_mcq(prompt, actions, use_chat=True)
	sorted_indices = np.argsort(probabilities)[::-1]
	print("Probabilites: ")
	for i in sorted_indices:
		print("{action}:\t{prob:10.3f}".format(action=actions[i], prob=probabilities[i]))
	
	# print(probabilities)
	1/0
	# video_path = "Videos/make_a_blt_0.mp4"
	# frame_range = [2161, 2327]
	# frame_range = [1356, 1600]
	# answer, best_frame = llm.SeViLa_VQA("What is the state of the tomato?", frame_range, video_path, num_responses = 10)
	# print("Answer: ", answer)
	# print("Best frame: ", best_frame)
	# 1/0
	# prompt_file = "/private/home/mverghese/prompt.txt"
	# result_file = "/private/home/mverghese/result.txt"
	# image_read_folder = "/private/home/mverghese/server_images"
	# server = Base_Server(prompt_file,result_file,image_read_folder)

	# # import threading
	# # prompt_thread = threading.Thread(target=server.prompt_listener)
	# # prompt_thread.start()

	# server.run_image_listener()

	# # interface = transformers_interface()
	# # prompt = ["this is a test prompt", "this is another test prompt"]
	# # print(interface.generate(prompt,num_tokens = 128))
	# # prompt_file = "socratic_prompt.txt"
	# # with open(prompt_file, 'r') as prompt_file:
	# #     prompt = prompt_file.read()

	# # queries = ['put shoe', 'take shoe', 'give shoe', 'insert shoe', 'move shoe']
	# # print("Queries: ", queries)
	# # query_probs = interface.eval_log_probs_old(prompt, queries, verbose = True)
	# # print(query_probs)
	LLAMA_PATH = "/home/mverghese/Models/Llama-3.2-11B-Vision-Instruct/"
	llm = transformers_interface(LLAMA_PATH, cuda_devices=[0, 1], vision_model=True)
	messages = [
		{
			"role": "system",
			"content": "You are an assitant designed to help people with household tasks. Do not be verbose. Answer the question with no added qualifications or caveats. Just directly provide the answer.",
		},
		{
			"role": "user",
			"content": "What are the steps make a Bacon, Lettuce and Tomato sandwich. The ingredients are uncooked bacon, whole tomatoes, a whole head of lettuce, a loaf of sliced bread, and a jar of mayonnaise. List each step in a new line with a number. For example: 1. ",
		},
	]
	print(llm.tokenizer.apply_chat_template(messages, tokenize=False, return_tensors="pt", add_generation_prompt=True))
	1/0
	# prompt = "List the common steps to make a Bacon, Lettuce and Tomato sandwich. Separate the steps by commas."
	response = llm.generate_chat(messages, num_tokens=512)
	# response = llm.generate(prompt,num_tokens = 512)
	print(response)
	# yes_prob, no_prob = llm.yes_no_question(prompt)
	# print("Yes prob: {}, No prob: {}".format(yes_prob, no_prob))
	# output = llm.generate(prompt,num_tokens = 128)
	# print(output)
