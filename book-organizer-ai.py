import asyncio
import sys

import ebookmeta
from gpt_json import GPTJSON, GPTMessage, GPTMessageRole, GPTModelVersion
from pydantic import BaseModel
from pypdf import PdfReader

from keys import *


class BookSchema(BaseModel):
    title: str
    author_surnames: str
    has_edition: bool
    edition: int
    folder: str


SYSTEM_PROMPT = """
You are sorting eBooks (PDF, EPUB, etc.) into a series of folders describing their category, such as Computer Science, Social Sciences, Writing, Classics, Design, etc. Each folder can have subfolders describing subcategories, such as Computer Science/Algorithms, Social Sciences/Classics, Mathematics/Calculus & Analysis, etc. Books can also be classified by a single category if they are a general-topic book. 

You are also needed to specify the book's title and a comma-separated list of the surnames of the author(s) of the book. If the book has an edition, set "has_edition" to true and write the number of the edition. If the book has no edition, set "has_edition" to false and write -1 as the number of the edition.

For example:
Filename: Clifford Stein_ Thomas H. Cormen_ Ronald L. Rivest_ Charles Eric Leiserson - Introduction to algorithms (2022).pdf
Name: Introduction to Algorithms
Author: Cormen, Leiserson, Rivest, Stein
has_edition: True
edition: 4
Folder: Computer Science/Theoretical Computer Science/Algorithms & Data Structures

Filename: Crafting Interpreters by Robert Nystrom (z-lib.org).pdf
Name: Crafting Interpreters
Author: Nystrom
has_edition: False
edition: -1
Folder: Computer Science/Theoretical Computer Science/Programming Languages & Compilers

Filename: Principles of Modern Chemistry, 7e (Oxtoby, Gillis, Campion)
Name: Principles of Modern Chemistry
Author: Oxtoby, Gillis, Campion
has_edition: True
edition: 7
Folder: Chemistry

Given the filename of a book and its metadata, return the book's title, author surnames, category, and edition information.

Respond with the following JSON schema:

{json_schema}
"""

# Get the file name from the command line
file_name = sys.argv[1]
content = f"Filename: {file_name}\n"

if file_name.endswith("pdf"):
    # Read the file
    pdf_reader = PdfReader(file_name)
    # Print the metadata
    for key, value in pdf_reader.metadata.items():
        content += f"{key}: {value}\n"
elif file_name.endswith("epub"):
    meta = ebookmeta.get_metadata(file_name)
    if meta.title:
        content += f"Title: {meta.title}\n"
    if meta.author_list:
        content += f"Authors: {', '.join(meta.author_list)}\n"
    if meta.description:
        content += f"Description: {meta.description}\n"


async def runner(content):
    print(content)
    gpt_json = GPTJSON[BookSchema](API_KEY, model=GPTModelVersion.GPT_3_5)
    response, _ = await gpt_json.run(
        messages=[
            GPTMessage(
                role=GPTMessageRole.SYSTEM,
                content=SYSTEM_PROMPT,
            ),
            GPTMessage(
                role=GPTMessageRole.USER,
                content=content,
            ),
        ]
    )
    print(response)


asyncio.run(runner(content))
