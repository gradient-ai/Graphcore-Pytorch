"""
# Sentiment analysis with GPT2 using IPUs


The Generative Pre-trained Transformer 2 (GPT-2) model can be utilised for text generation, question answering, translation and summarisation. 

In this notebook, we will show you how to quickly utilise Pytorch and the IPU for fast sentiment analysis using our implementation of GPT-2 small and medium on an IPU-POD4, and GPT-2 Large on an IPU-POD16.

To run fine-tuning on your own dataset with GPT-2 on IPU, please follow our tutorial on this [add ref]. 

First let's install our requirements and build our custom ops:
"""
# !pip install -r requirements.txt
# !make

"""
## Text Generation using GPT-2 Small

GPT-2 Small is a 124M parameter model pre-trained on a vast corpus of English data, you can learn more about the model and the dataset used to train it on this [Hugging Face model card](https://huggingface.co/gpt2). 

Let's see how we can use the generative capabilities of GPT-2 to help us decide what to cook for dinner tonight with the following question:
"""
generation_prompt = "Question: What should we cook for dinner tonight?"
""""
GPT-2 Small has a small number of parameters meaning that it can be run on a single IPU, hence we must set options which are specific to work on this system configuration. 
Since we are not pipelining our model, we can set the majority of these settings to `None`. It is important to note here that:
 
 - `model_name_or_path` - allows us to refer to a specific GPT-2 model version, in this case we will be running inference on 'gpt2',
 - `single_ipu` - allows us to set specific defaults if running on only a single IPU, 
"""
from text_inference_pipeline import create_args

args = create_args("gpt2",
                    single_ipu=True,
                    layers_per_ipu=None,
                    matmul_proportion=None,
                    prompt = generation_prompt)

print(args)
"""
The `initialise_model` function loads and runs the model for one iteration on the input token, we can use that to see whats for dinner!
"""
from text_inference_pipeline import initialise_model

prompt, run_model, model, tokenizer = initialise_model(args)

print(prompt)
"""
From doing this small exercise, we have been able to see how quickly we can run and utilise GPT-2 to help us generate text!
The generated text is quite random and without context the model has been unable to correctly identify the task which we are trying to complete.

By focusing the task on Sentiment Analysis we should be able to  generate better results by providing context to the model through Few-Shot Learning.

## Sentiment analysis using Few-Shot learning

Using Few-Shot learning, we can guide GPT-2 to complete a more specific task by feeding the model context prior to the the input which we want the model to predict on.
For sentiment analysis, we can structure sentences in the following format: "Message: `text` Sentiment: `classification` ###".

The `text` will be a sentence with a `positive` , `negative` or `neutral` classification, the sentence finishes with `###` as the stop token.

Using this structure, we will feed the model with the context it needs to predict the next token, let's use the following sentences for context:
"""
neg = "Message: The weather has been horrible this winter... Sentiment: Negative ### "
pos = "Message: I love the IPU, it is so fast! Sentiment: Positive ### "
neutral = "Message: My family are coming to my house for dinner. Sentiment: Neutral ### "
"""
By concatenating these sentences together we have created the context needed to complete Few-Shot Learning on our model!
"""
few_shot_prompt = neg + pos + neutral
print(few_shot_prompt.replace("### ","\n"))
"""
Finally, we can now include a test prompt which we want to use to see if our model correctly predicts the next token in the sequence.
"""
test = "Message: That was the best movie I've seen this year! Sentiment:"
print((few_shot_prompt + test).replace("### ","\n"))
"""
The context in `few_shot_prompt` will be used for used for additional experiments later on, hence we will keep this separate from the `test` prompt which we will use to initialise and test our model.

# Fast inference on GPT-2 Small

Now with our `few_shot_prompt` for context and our `test` prompt for inference we can utilise GPT-2 Small for inference on the IPU!

Let's create a configuration to point towards these new prompts:
"""

args_small = create_args("gpt2",
                         single_ipu=True,
                         layers_per_ipu=None,
                         matmul_proportion=None,
                         prompt = few_shot_prompt + test)

print(args_small)
"""
The `initialise_model` function allows us to load and initialise the model on the IPU ready to quickly send the inputs from the host and receive the models outputs. 

Using this and the context which we put together earlier, we can now load our model and run sentiment analysis on our test prompt. 
"""
from text_inference_pipeline import initialise_model
# %% time
prompt, run_small_model, small_model, small_tokenizer = initialise_model(args_small)
print(prompt.replace("### ","\n")) 
"""
As we were hoping, our test token has been correctly classified to have a `Positive` sentiment!

Now that we have successfully initialised our model and run sentiment analysis on our test prompt, we are ready to build a function which should allow us to correctly identify any inputs.

The `sentiment_analysis` function below prompts the user to provide an input and fits that input into the correct format that we described above.
To provide the model with context, we will use the `few_shot_prompt` that we defined earlier and concatenate that before the user input. 
"""
from text_inference_pipeline import get_input

def sentiment_analysis(args, prompt, run_model, model, tokenizer):
    user_input = "### Message: " + input() + " Sentiment:"
    args.prompt = prompt + user_input
    text_ids, txt_len, input_len = get_input(tokenizer, args)
    model_output = run_model(text_ids, txt_len, model, tokenizer, input_len, args)
    output = model_output[len(prompt):]
    return output


"""
Now we can feed our own inputs to the model to run sentiment analysis on the fly!
"""
output = sentiment_analysis(args_small, few_shot_prompt, run_small_model, small_model, small_tokenizer)
print(output)
"""
# Using GPT-2 Medium 

GPT-2 Medium has 355M parameters, and is the next size up from the GPT-2 model.
This implementation has more decoder layers fitted to the model, which allows us to achieve more accurate results. 

Due to this increase in parameters, we must pipeline our model across 4 IPUs and set specific configuration options related to this larger model.
To learn more about pipelining models on the IPU, see our [tutorial on this topic](https://github.com/graphcore/tutorials/tree/master/tutorials/pytorch/pipelining).

We must set the following arguments to run GPT2 on 4 IPUs:
 - `model_name_or_path` must be changed to 'gpt2-medium',
 - `single_ipu` must now be set to `False`,
 - `layers_per_ipu` - specifies which of the model layers should be pipelined across the IPU, GPT-2 Medium is a 24 layer model which is split up across an IPU-POD4 with [1, 7, 8, 8] layers being placed on each chip respectively. 
 - `matmul_proportion` - allows us to control how much temporary memory is used when doing matrix multiplication and convolution operations on each chip, to learn more read our docs on [Available Memory Proportion](https://docs.graphcore.ai/projects/available-memory/en/latest/index.html). Since we are running on a POD4 we can the memory proportion for each chip as shown below.
 
"""
args_medium = create_args(model_name_or_path = 'gpt2-medium',
                          single_ipu = False,
                          layers_per_ipu = [1, 7, 8, 8],
                          matmul_proportion = [0.2, 0.2, 0.2, 0.2],
                          prompt = few_shot_prompt + test)

"""
Now that we've set these configuration parameters, we are now ready to initialise a new model to prepare it for fast inference on the IPU. 
"""
prompt, run_medium_model, medium_model, medium_tokenizer = initialise_model(args_medium)
"""
We can now run sentiment analysis on your own inputs using this model too!
"""
output = sentiment_analysis(args_medium, few_shot_prompt, run_medium_model, medium_model, medium_tokenizer)
print(output)
"""
# Using GPT2-Large on an IPU-POD16

GPT 2 Large is a 36 layer model which has 774M parameters, and should provide you with even better results than the previous two models. 
If you have access to an IPU-POD16, you can push the abilities of GPT2 further by running inference our implementation of GPT2-Large.

Since this is a much larger implementation, we must pipeline the 36 layers in the model across 16 IPUs in order to fit them within the memory constraints of the IPU, as well as tuning the `matmul_proportion` constraint.
The arguments below allow us to reset these parameters to fit this new configuration.
"""
args_large = create_args(model_name_or_path = 'gpt2-large',
                          single_ipu = False,
                          layers_per_ipu = [0, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2],
                          matmul_proportion = [0.2, 0.15, 0.2, 0.2, 0.2, 0.15, 0.15, 0.2, 0.2, 0.15, 0.2, 0.2, 0.2, 0.15, 0.15, 0.2],
                          prompt = few_shot_prompt + test)
"""
Again, in order to prepare the model ready for inference, we must initialising this model again. 
"""
prompt, run_model, model, tokenizer = initialise_model(args_large)

prompt, output = sentiment_analysis(args_large, few_shot_prompt, run_model, model, tokenizer)
print("Output:", output)
"""
# Conclusion

In this notebook we have seen how to quickly and easily use GPT-2 to run inference on user inputs for sentiment analysis!
We have also seen how to configure the IPU when scaling up our model up to larger GPT-2 implementations.

Detach!!!
"""
