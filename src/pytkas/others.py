import zipfile
import os

def create_zip(files_to_zip, zip_file_name):
    """
        Create a ZIP file based on list of files to be archived.

        Parameters
        ----------
        files_to_zip : List[string]
           List of files' paths to be archived

        zip_file_name : string

           Filename of the destination ZIP.

        Returns
        ---------

            Nothing, archive is created in the default directory
    """
    with zipfile.ZipFile(zip_file_name, 'w') as zipf:
        for file in files_to_zip:
            zipf.write(file, os.path.basename(file))
    print(f"ZIP file: {zip_file_name} containing {files_to_zip} generated successfully")



