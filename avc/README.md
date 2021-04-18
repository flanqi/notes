
### Implementation Guidelines
#### Writing Python Modules
- Do not use relative imports (except in `__init__.py`, we will discuss this in the Python package module).
- Don’t use `from module import *` (except in `__init__.py`, we will discuss this in thePython package module).
- Place executable scripts at the top-level of your directory structure.
- Place all imports at the top of the module in the following order: native Python libraries, 3rd party packages, local modules.
- Add paths to the `PYTHONPATH` env variable (not via `sys.path.append()`).
- Use `argparse` to define command-line options and arguments.

#### Logging
- Use at least three levels of logging in your code
  + Use logging level `DEBUG` for things you only want to log while developing.
  + Use logging level `INFO` for things you want to monitor while in production.
  + Use logging level `WARNING` for things that would need attention in production.
  + Use logging level `ERROR` for when an exception occurs .
- Configure your logging from logging configuration files to allow for easy switching between
development and production.
- Only configure your logger from the executing script.
- In modules, set up your logger as `logger = logging.getLogger(__name__)`.
- Don’t commit .log files to git, put .log in your `.gitignore`.
- Include timestamps in your logs.

#### Software Testing
- Do not write unit tests for any function that makes a network call (e.g. calls an API, queriesa database).
- Tests do not have to be written for plotting functions.
- Place tests in `tests/` folder.
- Structure the `tests/` folder the same as the code you’re testing (e.g. same structure as `src/`).
- For a module, `module_x.py`, name the test file, `test_module_x.py`.
- For all tests related to a `function()`, name the test function, `test_function_<descriptor>.py`, where `<descriptor>`
  describes what is being tested. 
- Create at least one “happy path” and one “unhappy path” test for each testable function for all class assignments 
  (this may not be sufficient in the real world).
- Use `pytest` for all class assignments.

#### Program Design and QA
- There should be NO hard coded variables in any code.
- Credentials and usernames should be provided as environment variables and never committed to version control.
- Variable names should be meaningful.
- Code should be structured for testability and debugging with modular functions and modules.
- Functions should do only one thing and be appropriately sized (unless they are orchestration functions, tying together 
multiple functions to do a larger task).
- Existing libraries should be leveraged where possible (rather than implementing functionality yourself).
- Code should be **PEP8-compliant**.
- Docstrings should be **PEP257-compliant**.


QA should explicitly address the following about the code being reviewed:
- Functionality
- Readability
- Design
- Testing
- Documentation

#### Creating and distributing Python packages
- Provide instructions for how to build in the README of your package
- When publishing to an artifact repository like PyPI:
  + Try to provide a built distribution
  + Do provide a source distribution
