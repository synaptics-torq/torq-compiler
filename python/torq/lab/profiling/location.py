import re
from typing import List
from dataclasses import dataclass

"""
This is a simple parser for MLIR location strings, based on the grammar defined in the MLIR docs

Unfortunately MLIR location attributes APIs do not seem to be exported by the MLIR Python bindings,
so we need to parse the location strings ourselves.
 
The parser is implemented as a recursive descent parser, and returns an AST representation of the location
that can be easily pretty printed or analyzed.
"""

@dataclass(frozen=True)
class LocationAST:
    """Base class for MLIR location AST nodes."""
    pass


@dataclass(frozen=True)
class UnknownLocation(LocationAST):
    """Represents an unknown location (?).
    
    Grammar:
        unknown-location ::= `?`
    """
    def __str__(self):
        return "?"


@dataclass(frozen=True)
class FileLineColLocation(LocationAST):
    """Represents a file:line:column location.
    
    Grammar:
        filelinecol-location ::= string-literal `:` integer-literal `:` integer-literal
    """
    file: str  # Includes quotes
    line: str
    col: str
    
    def __str__(self):
        return f"{self.file}:{self.line}:{self.col}"


@dataclass(frozen=True)
class CallsiteLocation(LocationAST):
    """Represents a callsite location.
    
    Grammar:
        callsite-location ::= `callsite` `(` location `at` location `)`
    
    Attributes:
        callee: The callee location
        caller: The caller location
    """
    callee: LocationAST
    caller: LocationAST
    
    def __str__(self):
        return f"callsite({self.callee} at {self.caller})"


@dataclass(frozen=True)
class FusedLocation(LocationAST):
    """Represents a fused location.
    
    Grammar:
        fusion-metadata ::= `<` attribute-value `>`
        fused-location ::= `fused` fusion-metadata? `[` (location (`,` location)* )? `]`
    
    Attributes:
        locations: List of fused locations
        metadata: Optional metadata string
    """
    locations: list
    metadata: str = None
    
    def __str__(self):
        locations_str = ", ".join(str(loc) for loc in self.locations)
        metadata_str = f"<{self.metadata}>" if self.metadata else ""
        return f"fused{metadata_str}[{locations_str}]"


@dataclass(frozen=True)
class NameLocation(LocationAST):
    """Represents a named location, optionally with a child location.
    
    Grammar:
        name-location ::= string-literal (`(` location `)`)?
    
    Attributes:
        name: The name (includes quotes)
        child: Optional child location
    """
    name: str  # Includes quotes
    child: LocationAST = None
    
    def __str__(self):
        if self.child:
            return f"{self.name}({self.child})"
        return self.name


def parse_location(loc_str: str) -> LocationAST:
        """
        Parse and return MLIR location AST according to grammar:
        
        top-level ::= `loc` `(` location `)`
        location ::= callsite-location | filelinecol-location | fused-location | name-location | unknown-location
        callsite-location ::= `callsite` `(` location `at` location `)`
        filelinecol-location ::= string-literal `:` integer-literal `:` integer-literal
        fusion-metadata ::= `<` attribute-value `>`
        fused-location ::= `fused` fusion-metadata? `[` (location (`,` location)* )? `]`
        name-location ::= string-literal (`(` location `)`)?
        unknown-location ::= `?` | `unknown`
        """
        
        input_str = str(loc_str)
        pos, tokens = [0], re.findall(r'"[^"]*"|\w+|[():<>,\[\]?]', input_str)
        
        def peek():
            return tokens[pos[0]] if pos[0] < len(tokens) else None
        
        def consume(expected=None): 
            if expected and peek() != expected:
                raise ValueError(f"Expected {expected}")
            
            pos[0] += 1

            return tokens[pos[0]-1]
        
        def parse_location_node():
            if peek() == "?" or peek() == "unknown":
                consume()
                return UnknownLocation()
            if peek() == "callsite": 
                consume()
                consume("(")
                callee = parse_location_node()
                consume("at")
                caller = parse_location_node()
                consume(")")
                return CallsiteLocation(callee, caller)
            if peek() == "fused":
                consume()
                metadata = None
                if peek() == "<":
                    consume("<")
                    metadata = consume()
                    consume(">")
                consume("[")
                locations = [parse_location_node()] if peek() != "]" else []
                while peek() == ",": 
                    consume(",")
                    locations.append(parse_location_node())
                consume("]")
                return FusedLocation(locations, metadata)
            if peek() and peek()[0] == '"':
                name = consume()
                if peek() == "(": 
                    consume("(")
                    child = parse_location_node()
                    consume(")")
                    return NameLocation(name, child)
                if peek() == ":": 
                    consume(":")
                    line = consume()
                    consume(":")
                    col = consume()
                    return FileLineColLocation(name, line, col)
                return NameLocation(name)
            raise ValueError(f"Unexpected: {peek()}")
        
        consume("loc")
        consume("(")
        result = parse_location_node()
        consume(")")
        
        return result

def extract_all_file_locations(loc: str) -> List[FileLineColLocation]:

    parsed_location = parse_location(loc)

    def get_file_locations(ploc: LocationAST):

            if isinstance(ploc, CallsiteLocation):
                return get_file_locations(ploc.callee) + get_file_locations(ploc.caller)
            elif isinstance(ploc, FileLineColLocation):
                return [ploc]
            elif isinstance(ploc, FusedLocation):
                result = []
                for floc in ploc.locations:
                    result += get_file_locations(floc)
                return result
            else:
                return []

    return get_file_locations(parsed_location)