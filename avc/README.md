
### Implementation Guidelines
#### Writing Python Modules
- [ ] Do not use relative imports (except in `__init__.py`, we will discuss this in the Python package module).
- [ ] Don’t use `from module import *` (except in `__init__.py`, we will discuss this in thePython package module).
- [ ] Place executable scripts at the top-level of your directory structure.
- [ ] Place all imports at the top of the module in the following order: native Python libraries, 3rd party packages, local modules.
- [ ] Add paths to the `PYTHONPATH` env variable (not via `sys.path.append()`).
- [ ] Use `argparse` to define command-line options and arguments.

#### Logging
- [ ] Use at least three levels of logging in your code
  + [ ] Use logging level `DEBUG` for things you only want to log while developing.
  + [ ] Use logging level `INFO` for things you want to monitor while in production.
  + [ ] Use logging level `WARNING` for things that would need attention in production.
  + [ ] Use logging level `ERROR` for when an exception occurs .
- [ ] Configure your logging from logging configuration files to allow for easy switching between
development and production.
- [ ] Only configure your logger from the executing script.
- [ ] In modules, set up your logger as `logger = logging.getLogger(__name__)`.
- [ ] Don’t commit .log files to git, put .log in your `.gitignore`.
- [ ] Include timestamps in your logs.

#### Software Testing
- [ ] Do not write unit tests for any function that makes a network call (e.g. calls an API, queriesa database).
- [ ] Tests do not have to be written for plotting functions.
- [ ] Place tests in `tests/` folder.
- [ ] Structure the `tests/` folder the same as the code you’re testing (e.g. same structure as `src/`).
- [ ] For a module, `module_x.py`, name the test file, `test_module_x.py`.
- [ ] For all tests related to a `function()`, name the test function, `test_function_<descriptor>.py`, where `<descriptor>`
  describes what is being tested. 
- [ ] Create at least one “happy path” and one “unhappy path” test for each testable function for all class assignments 
  (this may not be sufficient in the real world).
- [ ] Use `pytest` for all class assignments.

### Exception Handling
- [ ] Exception handling should always be used when calling APIs, databases, and loading files.
- [ ] Exception handilng should also be used to deal with unexpected input.
- [ ] Attempt to be as specific as possible for each possible exception type. Use the catch all `except`: as a final resort for errors that can not be expected ahead of time.
- [ ] Thoroughly read the documentation of the libraries you use to understand what exceptionsare thrown.
- [ ] Avoid `raise` if possible in any calling code that will be deployed as this will break the code and end the application.
- [ ] Use logging when possible instead of standard output like print statements.
- [ ] Limit the `try` clause to specific operations. For example, if you have to open a file andopen a database connection, then you should have two try-except blocks, one for the fileopening and one for the database connection.

#### Program Design and QA
- [ ] There should be NO hard coded variables in any code.
- [ ] Credentials and usernames should be provided as environment variables and never committed to version control.
- [ ] Variable names should be meaningful.
- [ ] Code should be structured for testability and debugging with modular functions and modules.
- [ ] Functions should do only one thing and be appropriately sized (unless they are orchestration functions, tying together 
multiple functions to do a larger task).
- [ ] Existing libraries should be leveraged where possible (rather than implementing functionality yourself).
- [ ] Code should be **PEP8-compliant**.
- [ ] Docstrings should be **PEP257-compliant**.


QA should explicitly address the following about the code being reviewed:
- [ ] Functionality
- [ ] Readability
- [ ] Design
- [ ] Testing
- [ ] Documentation

#### Creating and distributing Python packages
- [ ] Provide instructions for how to build in the README of your package
- [ ] When publishing to an artifact repository like PyPI:
  + [ ] Try to provide a built distribution
  + [ ] Do provide a source distribution

### Architechtural Considerations
- [ ] “Offline” inference means you will make predictions for a pre-determined set of possible input combinations and store them in your database (RDS) for later serving.
- [ ] Your webapp will require low latency so as you develop your model so the speed at which you can make an inference will determine what you need to do. 
- [ ] If inference takes a longtime you should implement a system architecture with “offline” inference.Collaborative will most likely require a system architecture with “offline” inference.

### Configurations
- [ ] All functions should have arguments (with limited exceptions).
- [ ] Credentials –> environment variables
- [ ] Environment-specific configurations –> environment variables
- [ ] Code configurations –> code or file
- [ ] Inputs –> command-line arguments and/or file
- [ ] Standard convention is to capitalize all environment variable names.
- [ ] Centralize reading of environment variables in one configuration script.

### Data Architectures
- [ ] Use row-based storage when you need to access entire records, such as when exposing auser’s information in an app.
- [ ] Use columnar-based storage when doing analytical queries, such as aggregations.
- [ ] Use object stores, like S3, for storing raw data (prior to any processing).



### Data Ingestion
- [ ] Use maps (dictionaries) to dedupe values
- [ ] Use maps (dictionaries) to retain lookup data when processing
  - [ ] When enriching a dataset with reference info
  - [ ] For small enough reference sets
- [ ] Use sets to establish uniqueness
- [ ] Use arrays (lists) to iterate through records
  - [ ] processing each field/record
  - [ ] validation each field/record
  - [ ] searching for a value in a small dataset
- [ ] Use a file format like Avro that handles schema evolution, for datasets that have changingschemas
- [ ] Partitioning large datasets with a partition key like date can improve query speeds and allow you to process subsets of a dataset in batches and/or concurrently
- [ ] Log out data quality/validation issues you encounter
- [ ] Use appropriate data types


### Effective Queries
- [ ] Put reserve keywords (e.g. SELECT, FROM) in all upper case.
- [ ] Make column names all lower case.
- [ ] Only select the columns you need.SELECT * degrades query performance.
- [ ] Also, if the table you query changes schema, the result of a SELECT * query will changeas well.
- [ ] Place each field in the SELECT, GROUP BY, ORDER BY statements on a separate line.
- [ ] Place table names in FROM, JOIN, etc on their own lines to make it easy to locate whattables are being used.

### Model Reproducibility
- [ ] Come up with a plan for versioning your code, configurations, and artifacts.
  - [ ] In this class, use YAML to version your machine learning pipeline parameters and configurations.
  - [ ] Store raw data and artifacts in S3.
  - [ ] Version code, YAML using git
- [ ] Document your machine learning pipeline workflow with Makefile or bash script.
- [ ] Always set, document, and version any random seeds used.
- [ ] Use Docker and a requirements.txt file to control your environment.
- [ ] Always run your model pipeline twice and compare artifacts to ensure reproducibility.

### EC2 Cloud Service
- [ ] Use long term persistence (e.g. S3 or RDS) when working with EC2 instances. Don’tjust keep your data on the server.
- [ ] If you do need a lot of fast read/writes of files, use EBS attached storage.
- [ ] By default, instances are not backed by EBS.Non EBS-backed machines cannot be “stopped”. They are either running or“terminated”. Any data that is not in EBS or long-term persistence like S3 or adatabase will be lost when you terminate the instance.
- [ ] For most workloads S3 makes more sense than EBS (slower, but cheaper).

### Feedbacks
- [ ] Please have final project Docker commands use a connection string, not the MYSQLenvironment variables
- [ ] README should have a standalone section enumerating/describing any env vars used inproject.
- [ ] misused whitespaces/hard tabs
- [ ] pylint for checking styles
