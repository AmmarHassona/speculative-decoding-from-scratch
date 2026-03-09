from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import random

# When GPT processes input, it outputs logits, a 3D tensor with the following shape:
# [batch_size, sequence_length, vocab_size]
# = [1, 5, 50257]  # 5 tokens input, 50257 possible next tokens

# dim=0 (rows) represents which batch (which input sequence)
# dim=1 (columns) represents which word in the vocabulary

# generate one token manually (greedy decoding)
# this is NOT the approach that speculative decoding follows, spec. decoding follows a resampling approach from the probability distribution 
# with torch.no_grad(): # disables gradient computation as we're doing inference not training
#     outputs = model(input_ids) # pass input tokens to model
#     logits = outputs.logits[:, -1, :] # [all_batches, last position in sequence, all vocab items]
#     print("Logits:", logits)

#     next_token = torch.argmax(logits, dim = -1) # this asks: for each row, which column has the highest value?
#                                                 # returns the highest-scoring word

def baseline_generate(model, tokenizer, prompt, max_tokens):
    input_ids = tokenizer(prompt, return_tensors = "pt").input_ids.to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens = max_tokens, do_sample = True)
    return tokenizer.decode(output[0])

def generate_draft(draft_model, input_ids, k):
    """
    A function that returns k number of tokens and their probabilities
    """
    # so we already have the model and the tokenized input, we need to loop and generate k number of tokens
    # torch.no_grad() is non iterable so i guess we can create an interable variable and use it in a while loop
    
    i = 0
    model_output = [] # array of output tokens
    draft_probs = [] # array of probabilities per token
    draft_probs_full = [] # array of probability distribuition for all tokens for residual sampling
    while i < k:
        with torch.no_grad(): # disables gradient computation as we're doing inference not training
            outputs = draft_model(input_ids) # pass tokenized input into draft model
            logits = outputs.logits[:, -1, :] # [all_batches, last position in sequence, all vocab items]
            probs = torch.softmax(logits, dim = -1) # convert logits to probabilities
                                                    # dim = -1 is the same as dim = 1 but its more flexible in case the tensor shape changes
            next_token = torch.multinomial(probs, num_samples = 1) # sample based on probability
            input_ids = torch.cat([input_ids, next_token], dim = 1) # concatenate the new token into the input id token sequence
            
            model_output.append(next_token) 
            token_prob = probs[0, next_token.squeeze()] # we want to store the prob of the token that was chosen not all the tokens, squeeze removes dimensions of size 1
            draft_probs_full.append(probs[0]) # store the full probability distribuition of all tokens
            draft_probs.append(token_prob)
            i += 1
    
    return model_output, draft_probs, draft_probs_full

def verify(target_model, input_ids_with_draft, prompt_len):
    """
    A function that returns target probabilities for all positions
    """

    with torch.no_grad():
        outputs = target_model(input_ids_with_draft)
        logits = outputs.logits[:, prompt_len-1:-1, :] # [batch_size, sequence_length, vocab_size]
                                                       # sequence_length is input_len - 1 because we want all positions up to the last position (exclusive)  
        target_probs = torch.softmax(logits, dim = -1)

    bonus_logits = outputs.logits[:, -1, :]
    bonus_probs = torch.softmax(bonus_logits, dim = -1) # best case scenario that all draft tokens are accepted so we can sample k+1 tokens
    
    return target_probs, bonus_probs

def rejection_sample(target_probs, draft_probs, draft_probs_full, draft_tokens):
    """
    A function that returns how many tokens to accept
    """
    # we need to compare the probs, if target is greater than or equal to draft, accept, else reject
    
    accepted_tokens = [] # we can use an array of booleans since we only need to store the state of that token and not the actual probability
    sampled_token = None # set sampled_token to none in case that the loop accepts all tokens

    for i in range(len(draft_probs)): 
        draft_token_id = draft_tokens[i].squeeze() # get the actual token id
        target_prob_for_token = target_probs[0, i, draft_token_id]  # prob target assigns to that token
        draft_prob = draft_probs[i]

        acceptance_prob = min(1.0, (target_prob_for_token / draft_prob).item())

        if random.random() < acceptance_prob: # the reason we use random here is because we want the sampling to be probabilistic not deterministic
                                              # what this means is that if we specify a certain threshold, it prevents real sampling and relies on passing the threshold
                                              # this is called Bernoulli Sampling, and it is the approach that the original paper follows
            accepted_tokens.append(True) # append true if accept
        else:
            # we need to subtract the draft probability distribution from the target probability distribution to get the residual
            residual = target_probs[0, i, :] - draft_probs_full[i]
            residual = torch.clamp(residual, min = 0) # clamp residual if value is negative to 0
            residual = residual / residual.sum() # normalize to sum to 1
            sampled_token = torch.multinomial(residual, num_samples = 1)
            break

    return accepted_tokens, sampled_token

def speculative_decoding(draft_model, target_model, tokenizer, prompt, k, max_tokens):
    input_ids = tokenizer(prompt, return_tensors = "pt").input_ids.to(device)
    # initialize variable to track acceptance rate of speculative decoding
    total_drafted = 0 
    total_accepted = 0

    generated_tokens = 0
    while generated_tokens < max_tokens:
        # generate draft tokens
        draft_output, draft_probs, draft_probs_full = generate_draft(draft_model, input_ids, k)

        draft_tokens = torch.cat(draft_output, dim = 1) # convert draft output to a tensor first as it is a list
        input_ids_with_draft = torch.cat([input_ids, draft_tokens], dim = 1) # concatenate the draft tokens into the input sequence

        # verify with target model
        prompt_len = input_ids.shape[1]
        target_probs, bonus_probs = verify(target_model, input_ids_with_draft, prompt_len)

        # rejection sample and append accepted tokens into input_ids
        accepted_tokens, sampled_token = rejection_sample(target_probs, draft_probs, draft_probs_full, draft_output)
       
        for i in range(len(accepted_tokens)): 
            token = draft_output[i] # slice draft output with the position of the accepted token
            input_ids = torch.cat([input_ids, token], dim = 1)

        total_drafted += k
        total_accepted += len(accepted_tokens)

        generated_tokens += len(accepted_tokens)

        # residual sampling of token, if token tokens are rejected we apply residual sampling else best case
        if sampled_token is not None:
            input_ids = torch.cat([input_ids, sampled_token.unsqueeze(0)], dim = 1)
            generated_tokens += 1
        elif len(accepted_tokens) == k:
        # best case: if all tokens are accepted sample one bonus from
            bonus = torch.multinomial(bonus_probs, num_samples = 1)
            input_ids = torch.cat([input_ids, bonus], dim = 1)
            generated_tokens += 1

    acceptance_rate = total_accepted / total_drafted
    return tokenizer.decode(input_ids[0]), acceptance_rate

def benchmark(draft_model, target_model, tokenizer, prompt, k, max_tokens, draft_name="draft", target_name="target"):
    print(f"\n{'='*50}")
    print(f"Prompt: '{prompt}'")
    print(f"k={k}, max_tokens={max_tokens}")
    print(f"{'='*50}")

    # draft model baseline
    start = time.time()
    draft_baseline_result = baseline_generate(draft_model, tokenizer, prompt, max_tokens)
    draft_baseline_time = time.time() - start
    draft_baseline_tps = max_tokens / draft_baseline_time
    print(f"\nDraft model ({draft_name}) baseline:")
    print(f"  Time: {draft_baseline_time:.2f}s")
    print(f"  Tokens/sec: {draft_baseline_tps:.1f}")
    print(f"  Output: {draft_baseline_result[:80]}")

    # target model baseline
    start = time.time()
    target_baseline_result = baseline_generate(target_model, tokenizer, prompt, max_tokens)
    target_baseline_time = time.time() - start
    target_baseline_tps = max_tokens / target_baseline_time
    print(f"\nTarget model ({target_name}) baseline:")
    print(f"  Time: {target_baseline_time:.2f}s")
    print(f"  Tokens/sec: {target_baseline_tps:.1f}")
    print(f"  Output: {target_baseline_result[:80]}")

    # speculative decoding
    start = time.time()
    spec_result, acceptance_rate = speculative_decoding(draft_model, target_model, tokenizer, prompt, k, max_tokens)
    spec_time = time.time() - start
    spec_tps = max_tokens / spec_time
    print(f"\nSpeculative decoding ({draft_name} -> {target_name}):")
    print(f"  Time: {spec_time:.2f}s")
    print(f"  Tokens/sec: {spec_tps:.1f}")
    print(f"  Acceptance rate: {acceptance_rate:.1%}")
    print(f"  Speedup vs target baseline: {spec_tps / target_baseline_tps:.2f}x")
    print(f"  Output: {spec_result[:80]}")
    print(f"{'='*50}\n")

if __name__ == "__main__":

    device = "mps"
    
    # draft_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)
    # target_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium").to(device)
    # tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    # draft_name = "gpt2"
    # target_name = "gpt2-medium"

    # draft_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B").to(device)
    # target_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B").to(device)
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    # draft_name = "Qwen2-0.5B"
    # target_name = "Qwen2-1.5B"

    draft_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B").to(device)
    target_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B").to(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    draft_name = "Qwen3.5-0.8B"
    target_name = "Qwen3.5-2B"

    prompts = [
        ("conversational", "Why did the chicken cross the road"),
        ("code",           "def fibonacci(n):"),
        ("factual",        "The theory of relativity states that"),
        ("repetitive", "1, 2, 3, 4, 5, 6, 7, 8, 9,"),
    ]

    k_values = [1, 2, 4, 6, 8]
    max_tokens = 50
    runs_per_config = 2
    output_file = f"results_{draft_name}_vs_{target_name}.txt"

    with open(output_file, "w") as f:

        def log(text=""):
            print(text)
            f.write(text + "\n")

        log(f"{'='*60}")
        log(f"Model pair:      {draft_name} -> {target_name}")
        log(f"Device:          {device}")
        log(f"max_tokens:      {max_tokens}")
        log(f"runs_per_config: {runs_per_config}")
        log(f"{'='*60}")

        # ── acceptance rate by prompt type (k=4) ──────────
        log("\n\n>>> ACCEPTANCE RATE BY PROMPT TYPE (k=4)\n")

        for prompt_type, prompt in prompts:
            log(f"\n--- Prompt type: {prompt_type} ---")
            log(f"Prompt: '{prompt}'")

            acceptance_rates = []
            speedups = []

            for run in range(1, runs_per_config + 1):
                start = time.time()
                baseline_generate(target_model, tokenizer, prompt, max_tokens)
                baseline_tps = max_tokens / (time.time() - start)

                start = time.time()
                _, acceptance_rate = speculative_decoding(draft_model, target_model, tokenizer, prompt, 4, max_tokens)
                spec_tps = max_tokens / (time.time() - start)

                acceptance_rates.append(acceptance_rate)
                speedups.append(spec_tps / baseline_tps)
                log(f"  Run {run}: acceptance={acceptance_rate:.1%}  speedup={spec_tps/baseline_tps:.2f}x")

            avg_acc = sum(acceptance_rates) / len(acceptance_rates)
            avg_spd = sum(speedups) / len(speedups)
            log(f"  AVG: acceptance={avg_acc:.1%}  speedup={avg_spd:.2f}x")

        # ── k ablation (conversational prompt) ────────────
        log("\n\n>>> K ABLATION STUDY\n")
        prompt_type, prompt = prompts[0]
        log(f"Prompt: '{prompt}'\n")

        for k in k_values:
            log(f"--- k={k} ---")

            acceptance_rates = []
            speedups = []

            for run in range(1, runs_per_config + 1):
                start = time.time()
                baseline_generate(target_model, tokenizer, prompt, max_tokens)
                baseline_tps = max_tokens / (time.time() - start)

                start = time.time()
                _, acceptance_rate = speculative_decoding(draft_model, target_model, tokenizer, prompt, k, max_tokens)
                spec_tps = max_tokens / (time.time() - start)

                acceptance_rates.append(acceptance_rate)
                speedups.append(spec_tps / baseline_tps)
                log(f"  Run {run}: acceptance={acceptance_rate:.1%}  speedup={spec_tps/baseline_tps:.2f}x")

            avg_acc = sum(acceptance_rates) / len(acceptance_rates)
            avg_spd = sum(speedups) / len(speedups)
            log(f"  AVG: acceptance={avg_acc:.1%}  speedup={avg_spd:.2f}x\n")

    print(f"\nResults written to {output_file}")