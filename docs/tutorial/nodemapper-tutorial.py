# %% [markdown]
# # NodeMapper tutorial
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/angelphanth/hugging-mapper/blob/main/docs/tutorial/nodemapper-tutorial.ipynb)
#
# In this demonstration we learn how we can use the `NodeMapper` class to
# - create a map from ids to text embeddings and 
# - perform similarity search using huggign face models.
#
# `NodeMapper` extends `HuggingMapper` by allowing you to:
# - Find similar nodes based on their texts, returning associated ids
# - Retrieve the best match or top-k matches for a given input
# - Visualize the embeddings in a tsne
#
# Let's get started!

# %%
# uncomment if colab
# #!pip install pandas hugging-mapper

# %% [markdown]
# First we generate demo data for the tutorial

# %%
import pandas as pd

# An example dataframe
# generate data
ids = ["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8"]
texts = [
    "They are happy",
    "I would like to order a doughnut",
    "The grass is green",
    "They are sad",
    "Have you poured the foundation?",
    "I am feeling grey",
    "blue",
    "home",
]
# to dataframe
df = pd.DataFrame({"id": ids, "text": texts})

# %% [markdown]
# Initializing `NodeMapper` will
# - load the given huggingface model
# - generate embeddings for the text column
# - creating a dictionary of the node ids : text embeddings
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
from hugger.mapper import NodeMapper
# init
mapper = NodeMapper(
    df=df,
    text_col="text",
    id_col="id",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# %% [markdown]
# Like `HuggingMapper` can simply get embeddings for given text

# %%
# generate embedding for a single text
embedding = mapper.embed_text("Good morning")
print(embedding.shape)

# generate embeddings for a list of texts
embeddings = mapper.embed_text(["Hello world", "Good evening", "Lunch time!"])
print(embeddings.shape)

# %% [markdown]
# But the main purpose of `NodeMapper` is to query for similar texts and their corresponding ids given a text input

# %%
# retrieve those most similar to given text, above threshold
mapper.get_similar("concrete", threshold=0)  # threshold 0 returns all

# %%
# retrieve top match, above threshold
print(mapper.get_match("joyful", threshold=0.4), "\n")
print(mapper.get_match("we are crying", threshold=0.4), "\n")
print(mapper.get_match("eatting a donut", threshold=0.4), "\n")

# %%
# retrieve top k matches, above threshold
print(mapper.get_similar("yellow", threshold=0.3, top_k=2), "\n")
print(mapper.get_similar("laughter", top_k=3), "\n")
