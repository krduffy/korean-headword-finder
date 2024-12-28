from typing import List, Literal, Tuple

# ( example text, source, index of correct sense )
type Example = Tuple[str, str, int]

# ( example texts, list of senses )
type TestCase = Tuple[List[Example], List[str]]

type Language = Literal["english", "korean"]
