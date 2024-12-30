from typing import List, Literal, Tuple

# ( example text, source, index of correct sense )
type Example = Tuple[str, str, int]

# ( example texts, list of senses )
type TestCaseForMatchingSenses = Tuple[List[Example], List[str]]

# ( list of lists of examples of usage, list of examples with unknown usage )
type TestCaseForMatchingUsage = Tuple[List[List[str]], List[str]]

type Language = Literal["english", "korean"]
