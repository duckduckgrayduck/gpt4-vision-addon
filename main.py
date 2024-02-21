"""
DocumentCloud Add-On that allows you to 
pull tabular information from documents with GPT4-Vision
"""
import os
import csv
import json
import zipfile
from typing import Annotated, Any, List
from io import StringIO
from documentcloud.addon import AddOn
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


class Vision(AddOn):
    """Extract tabular data with GPT4-Vision"""

    def main(self):
        """The main add-on functionality goes here."""
        default_prompt_text = """
            First take a moment to reason about the best set of headers for the tables.
            Write a good h1 for the image above. Then follow up with a short description of the what the data is about.
            Then for each table you identified, write a h2 tag that is a descriptive title of the table.
            Then follow up with a short description of the what the data is about.
            Lastly, produce the markdown table for each table you identified.
            Make sure to escape the markdown table properly, and make sure to include the caption and the dataframe.
            including escaping all the newlines and quotes. Only return a markdown table in dataframe, nothing else.
            """
        client = instructor.patch(OpenAI(api_key=os.environ["TOKEN"]), mode=instructor.function_calls.Mode.MD_JSON)
        prompt = self.data.get("prompt", default_prompt_text)
        output_format = self.data.get("output_format", "csv")
        page = self.data.get("page", 1)

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

        def save_tables_to_json(tables, filename):
            with open(filename, "w", encoding="utf-8") as jsonfile:
                json.dump(tables, jsonfile, indent=4, cls=TableEncoder)

        def save_tables_to_csv(tables, filename):
            with open(filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                for table in tables:
                    writer.writerow([table.caption])
                    writer.writerow([])
                    writer.writerows(table.dataframe.values.tolist())
                    writer.writerow([])  # Add empty rows between tables

        def md_to_df(data: Any) -> Any:
            if isinstance(data, str):
                return (
                    pd.read_csv(
                        StringIO(data),  # Get rid of whitespaces
                        sep="|",
                        index_col=1,
                    )
                    .dropna(axis=1, how="all")
                    .iloc[1:]
                    .map(lambda x: x.strip())
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
                                "text": f"{prompt}",
                            },
                        ],
                    }
                ],
            )

            return tables

        """for document in self.get_documents():
            image_url = document.get_large_image_url(page)
            tables = extract(image_url)
            if output_format == "csv":
                save_tables_to_csv(tables.tables, f"tables-{document.id}.csv")
                self.upload_file(open("tables.csv"))
            if output_format == "json":
                save_tables_to_json(tables.tables, f"tables-{document.id}.json")
                self.upload_file(open("tables.json"))"""

        zip_filename = "all_tables.zip"
        zipf = zipfile.ZipFile(zip_filename, "w")  # Create a zip file
        created_files = []  # Store the filenames of the created files

        for document in self.get_documents():
            image_url = document.get_large_image_url(page)
            tables = extract(image_url)
            if output_format == "csv":
                csv_filename = f"tables-{document.id}.csv"
                save_tables_to_csv(tables.tables, csv_filename)
                zipf.write(csv_filename)
                created_files.append(csv_filename)
                os.remove(csv_filename)  # Remove the CSV file after adding it to the zip
            if output_format == "json":
                json_filename = f"tables-{document.id}.json"
                save_tables_to_json(tables.tables, json_filename)
                zipf.write(json_filename)
                created_files.append(json_filename)
                os.remove(json_filename)  # Remove the JSON file after adding it to the zip

        zipf.close()  # Close the zip file

        # Upload the zip file
        with open(zip_filename, "rb") as f:
            self.upload_file(f)


if __name__ == "__main__":
    Vision().main()
