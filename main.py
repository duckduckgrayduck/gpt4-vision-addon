"""
DocumentCloud Add-On that allows you to 
pull tabular information from documents with GPT4-Vision
"""
import os
import sys
import csv
import json
import zipfile
from typing import Annotated, Any, List
from io import StringIO
from documentcloud.addon import AddOn
from documentcloud.exceptions import APIError
from openai import OpenAI
from pydantic import (
    BaseModel,
    BeforeValidator,
    PlainSerializer,
    InstanceOf,
    WithJsonSchema,
)
import instructor
import pandas as pd

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


class Vision(AddOn):
    """Extract tabular data with GPT4-Vision"""

    def calculate_cost(self, documents):
        """ Given a set of documents, counts the number of pages and returns a cost"""
        total_num_pages = 0
        for doc in documents:
            start_page = self.data.get("start_page", 1)
            end_page = self.data.get("end_page")
            last_page = 0
            if end_page <= doc.page_count:
                last_page = end_page
            else:
                last_page = doc.page_count
            pages_to_analyze = last_page - start_page + 1
            total_num_pages += pages_to_analyze
        cost = total_num_pages
        print(cost)
        return cost


    def validate(self):
        """Validate that we can run the analysis"""

        if self.get_document_count() is None:
            self.set_message(
                "It looks like no documents were selected. Search for some or "
                "select them and run again."
            )
            sys.exit(0)
        if not self.org_id:
            self.set_message("No organization to charge.")
            sys.exit(0)
        ai_credit_cost = self.calculate_cost(
            self.get_documents()
        )
        try:
            self.charge_credits(ai_credit_cost)
        except ValueError:
            return False
        except APIError:
            return False
        return True

    def main(self):
        """The main add-on functionality goes here."""
        self.client.session.headers.update({'User-Agent': 'GPT 4 Vision Add-On'})
        if not self.validate():
            self.set_message("You do not have sufficient AI credits to run this Add-On on this document set")
            sys.exit(0)
        default_prompt_text = """
            Take a moment to reason about the best set of headers for the tables.
            Write a good h1 for the image above. Then follow up with a short description of the what the data is about.
            Then for each table you identified, write a h2 tag that is a descriptive title of the table.
            Then follow up with a short description of the what the data is about.
            Lastly, produce the markdown table for each table you identified.
            Make sure to escape the markdown table properly, and make sure to include the caption and the dataframe.
            including escaping all the newlines and quotes. Only return a markdown table in dataframe, nothing else.
            """
        client = instructor.patch(
            OpenAI(api_key=os.environ["TOKEN"]), mode=instructor.function_calls.Mode.MD_JSON
        )
        prompt = self.data.get("prompt", "")
        final_prompt = prompt + "\n" + default_prompt_text
        output_format = self.data.get("output_format", "csv")
        start_page = self.data.get("start_page", 1)
        end_page = self.data.get("end_page", 1)

        if end_page < start_page:
            self.set_message("The end page you provided is smaller than the start page, try again")
            sys.exit(0)
        if start_page < 1:
            self.set_message("Your start page is less than 1, please try again")
            sys.exit(0)

        class TableEncoder(json.JSONEncoder):
            """ Used to transform dataframe -> JSON 
                output for save_tables_to_json
            """
            def default(self, o):
                if isinstance(o, Table):
                    cleaned_data = {}
                    for key, value in o.dataframe.to_dict().items():
                        cleaned_key = key.strip()
                        cleaned_values = {
                            sub_key.strip(): sub_value
                            for sub_key, sub_value in value.items()
                        }
                        cleaned_data[cleaned_key] = cleaned_values
                    return {
                        "caption": o.caption,
                        "dataframe": cleaned_data,
                    }
                return super().default(o)

        def save_tables_to_json(tables, json_file, page_number):
            with open(json_file, "a", encoding="utf-8") as jsonfile:  # Append mode
                jsonfile.write(f"Page number: {page_number}")
                json.dump(tables, jsonfile, indent=4, cls=TableEncoder)
                jsonfile.write('\n')
                jsonfile.write('\n')
                jsonfile.write('\n')

        def save_tables_to_csv(tables, csv_file, page_number):
            with open(csv_file, "a", newline="", encoding="utf-8") as csvfile:  # Append mode
                writer = csv.writer(csvfile)
                writer.writerow([f"Page Number: {page_number}"])  # Write the page number
                for table in tables:
                    writer.writerow([table.caption])
                    writer.writerows(table.dataframe.values.tolist())
                    writer.writerow([])  # Add empty rows between tables
                    writer.writerow([])
                    writer.writerow([])

        def md_to_df(data: Any) -> Any:
            if isinstance(data, str):
                return (
                    pd.read_csv(
                        StringIO(data),  # Get rid of whitespaces
                        sep="|",
                        index_col=None,
                    )
                    .dropna(axis=1, how="all")
                    .iloc[1:]
                    .applymap(lambda x: x.strip() if isinstance(x, str) else x)
                )
            return data

        MarkdownDataFrame = Annotated[
            InstanceOf[pd.DataFrame],
            BeforeValidator(md_to_df),
            PlainSerializer(lambda x: x.to_markdown()),
            WithJsonSchema(
                {
                    "type": "string",
                    "description": """
                        The markdown representation of the table,
                        each one should be tidy, do not try to join tables
                        that should be seperate""",
                }
            ),
        ]

        class Table(BaseModel):
            """Where we define a table"""

            caption: str
            dataframe: MarkdownDataFrame

        class MultipleTables(BaseModel):
            """Where we define multiple tables"""

            tables: List[Table]

        example = MultipleTables(
            tables=[
                Table(
                    caption="This is a caption",
                    dataframe=pd.DataFrame(
                        {
                            "Chart A": [10, 40],
                            "Chart B": [20, 50],
                            "Chart C": [30, 60],
                        }
                    ),
                )
            ]
        )

        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
        def extract(url: str) -> MultipleTables:
            tables = client.chat.completions.create(
                model="gpt-4-vision-preview",
                max_tokens=4000,
                response_model=MultipleTables,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this data accurately as a table"
                                f" in markdown format. {example.model_dump_json(indent=2)}",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": url},
                            },
                            {
                                "type": "text",
                                "text": f"{final_prompt}",
                            },
                        ],
                    }
                ],
            )

            return tables

        zip_filename = "all_tables.zip"
        zipf = zipfile.ZipFile(zip_filename, "w")  # Create a zip file
        created_files = []  # Store the filenames of the created files

        for document in self.get_documents():
            outer_bound = end_page + 1
            if end_page > document.page_count:
                outer_bound = document.page_count + 1
            if output_format == "csv":
                csv_filename = f"tables-{document.id}.csv"
                for page_number in range(start_page, outer_bound):
                    image_url = document.get_large_image_url(page_number)
                    tables = extract(image_url)
                    save_tables_to_csv(tables.tables, csv_filename, page_number)
                zipf.write(csv_filename)
                created_files.append(csv_filename)
            elif output_format == "json":
                json_filename = f"tables-{document.id}.json"
                for page_number in range(start_page, end_page + 1):
                    image_url = document.get_large_image_url(page_number)
                    tables = extract(image_url)
                    save_tables_to_json(tables.tables, json_filename, page_number)
                zipf.write(json_filename)
                created_files.append(json_filename)

        zipf.close()  # Close the zip file

        # Upload the zip file
        with open(zip_filename, "rb") as f:
            self.upload_file(f)


if __name__ == "__main__":
    Vision().main()
