# %% [markdown]
# # HuggingMapper Tutorial
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/angelphanth/hugging-mapper/blob/main/docs/tutorial/huggingmapper-tutorial.ipynb)
#
# In this notebook, we demo how to use the `HuggingMapper` class to generate text embeddings using state-of-the-art Hugging Face transformer models. 
#
# The `HuggingMapper` provides a simple interface for loading transformer models, tokenizing text, and extracting normalized embeddings.
#
# Here we:
# - Initializing the `HuggingMapper` with a pre-trained model
# - Generating embeddings for individual texts 
# - and lists of texts
#
# Let's get started!

# %%
# uncomment if colab
# #!pip install pandas hugging-mapper

# %% [markdown]
# Calling `HuggingMapper` will instantly load the given huggingface model
#
# <details>
# <summary style=color:orange> 
# <u>Click here</u> for more info on: <br><i><b>"Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads."</i></b>
# </summary>
# <h1></h1>
#
# **What is an HF_TOKEN / huggingface user access token?**
# > *[User Access Tokens are the preferred way to authenticate an application or notebook to Hugging Face services. You can manage your access tokens in your settings.](huggingface.co/docs/hub/en/security-tokens#what-are-user-access-tokens)*
#
# If you have an `HF_TOKEN` you can add it to your environment variables, repository secrets, and/or you can access it in your venv by saving the HF_TOKEN in an *.env* file and then loading it via package [`python-dotenv`](https://pypi.org/project/python-dotenv/).
#
# For example: 
# 1. More information for getting a Huggingface user token: [their docs](https://huggingface.co/docs/hub/en/security-tokens)
# 2. Save to "HF_TOKEN" variable
#
#     Example *.env* file: 
#     ```bash
#     HF_TOKEN=hf***...
#     ```
# 3. Access the .env variables via python-dotenv 
#
#     e.g. 
#     ```python
#     from dotenv import load_dotenv
#     load_dotenv()
#     ```
#
# <h1></h1>
# </details>

# %%
from hugger.mapper import HuggingMapper

# init
mapper = HuggingMapper(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# %% [markdown]
# Get embeddings for given text

# %%
# generate embedding for a single text
embedding = mapper.embed_text("Good morning")
print(embedding.shape)

# generate embeddings for a list of texts
embeddings = mapper.embed_text(["Hello world", "Good evening", "Lunch time!"])
print(embeddings.shape)
