from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import Optional
import requests,json
load_dotenv()

class Output(BaseModel):
    step: str=Field(...,description="The step which will be displayed to the user")
    content: Optional[str]=Field(None,description="The content of the step")
    tool: Optional[str]=Field(None,description="The tool which will be displayed to the user")
    input: Optional[str]=Field(None,description="The input of the step")

client=OpenAI()

SYSTEM_PROMPT = """
    You are an expert AI assistant in resolving user queries using chain of thoughts.
    You work on START, PLAN and OUTPUT steps.
    You need first to PLAN what needs to be done. The PLAN can be multiple steps.
    Once you think enough about PLAN, finally you execute it and give an OUTPUT. You can call a tool if required 
    from the llist of available tools. After calling tool, wait for the observe step which is the output of the tool.

    Rules:
    -The sequence of steps is START (when user gives an input), PLAN and finally OUTPUT( which is going to be displayed
    to the user).
    -Strictly follow the given JSON output format.
    
    Available Tools:
    -get_weather : Takes city name as an input string and returns the weather info about the city.

    Output JSON Format:
    {"step":"START"|"PLAN"|"OUTPUT"|"TOOL",'content':'string','tool':'string',"input":"string"}

    Example1:
    START:Hey can you solve 2+3*5/10?
    PLAN : {'step':"PLAN":'content':"seems like user want to solve algebra problem"}
    PLAN : {'step':"PLAN":'content':"we should solve this problem using BODMAS method"}
    PLAN : {'step':"PLAN":'content':"Yes, the BODMAS is correct thing to be done here"}
    PLAN : {'step':"PLAN":'content':"first we must multiply 3*5 which is 15"}
    PLAN : {'step':"PLAN":'content':"Now the equation becomes 2+15/10"}
    PLAN : {'step':"PLAN":'content':"we need to divide 15 by 10 which is 1.5"}
    PLAN : {'step':"PLAN":'content':"now the equation becomes 2+1.5"}
    PLAN : {'step':"PLAN":'content':"the addition gives 3.5"}
    PLAN : {'step':"PLAN":'content':"Great, we solved the problem and the answer is 3.5}
    OUTPUT:{"step":"OUTPUT":'content':"3.5"}
    
    Example2:
    START:What is the weather of New York City?
    PLAN : {'step':"PLAN":'content':"seems like user want to know the weather of new york city"}"}
    PLAN : {'step':"PLAN":'content':"let me check whether I have any available tool from the list of available tools"}
    PLAN : {'step':"PLAN":'content':"Great! there is get_weather tool available for this query"}
    PLAN : {'step':"PLAN":'content':"I need to call this tool for New York City as an input"}"}
    PLAN : {'step':"TOOL":'tool':"get_weather",'input':"New York City"}
    PLAN : {'step':"OBSERVE":'tool':'get_weather',"output":"Then current temperature of New York city is 21 degree Celcius. sky is clean without any sign of cloud. "}
    PLAN : {'step':"PLAN":"content":"Great. i have got the right info!"}
    OUTPUT:{"step":"OUTPUT":'content':"The current temperature of New York city is 21 degree Celcius. sky is clean without any sign of cloud. "}
    
    Example3:
    START:How can you prove the statement that "every integral domain with unique factorisation domain
     is a Dedekind domain?
    PLAN : {'step':"PLAN":'content':'seems like user want to know the relation that every integral domain 
    with unique factorisation domain implies a Dedekind domain'}
    PLAN : {'step':"PLAN":'content':'If any integral domain is a unnique factorisation domain, 
    every ideal is a finite product of prime ideals.'}
    PLAN : {'step':"PLAN":'content':'Since every ideal is a finite product of prime ideals, this domain
    satisfies acsending chain condition, and therefore it is Noetherian.'}
    PLAN : {'step':"PLAN":'content':'Since it is Noetherian, if its dimension is less than or equal to 1, it is a 
    Dedekind domain'}
    PLAN : {'step':"PLAN":'content':'In order to prove the dimension is less than or equal to 1, assume the contrary.'} 
    PLAN : {'step':"PLAN":'content':'Then there exist two nonzero prime ideals that make ascending chain of prime ideals.'}    
    PLAN : {'step':"PLAN":'content':'Then one prime contain the other, but both are primes, which is a contradiction'}    
    PLAN : {'step':"PLAN":'content':'Contradiction gives that dimension of this integral domain is less than or equal to 1.'}
    PLAN : {'step':"PLAN":'content':'Since it is Noetherian and dimension is less than or equal to 1, it is a 
    Dedekind domain'}
    OUTPUT: {'step':"OUTPUT":'content':"proved"}
    
    Example4:
    START:How can you prove the statement that "for the two sequences of locally compact spaces with direct limits, 
    the cartesian product topology on the product of two direct limits coincide with the direct limit topology which is associatd with
    the sequence of products"?
    PLAN : {'step':"PLAN":'content':'seems like user want to know two topologies,one induced by the direct limit of the product of two sequnces of locally
    compact spaces and the other induced by the product of the direct limits of two sequences are equivalent.'}
    PLAN : {'step':"PLAN":'content':"Consider an open set in the direct limit topology, and pick an arbitrary point of that set."}
    PLAN : {'step':"PLAN":'content':"By the definition of direct limit, there exists an integer i such that that point is in the product of i-th two locally compact spaces."}
    PLAN : {'step':"PLAN":'content':"Then I can choose compact neighborhoods of that point in the first coordinate, and in the second coordinate respectively such that the product of
    those compact neighborhood is contained in the open set I chose above in the direct limit topology."}
    PLAN : {'step':"PLAN":'content':"I can repeat this process for i+1-th sequence and so on."}
    PLAN : {'step':"PLAN":'content':"By induction, I can get two sequences of compact neighborhood and I can define the unions of each sequences."}
    PLAN : {'step':"PLAN":'content':"Then those two unions are open sets and the point I picked is contained in the product of those open sets."}
    PLAN : {'step':"PLAN":'content':"Then those two unions are open sets is contained in the open set I picked in the direct limit topology.."}
    PLAN : {'step':"PLAN":'content':"Therefore the open set we chose in the direct limit topology is open in the product topology of two sequences."}
    OUTPUT:{'step':"OUTPUT":'content':"proved"}



"""


def get_weather(s):
    url=f"https://wttr.in/{s.lower()}?format=%C+%t"
    response=requests.get(url)
    if response.status_code==200:
        return f"{response.text}"
    return "something went wrong"
message_history=[   {'role':'system','content':SYSTEM_PROMPT},
]
available={'get_weather':get_weather}
user_query=input("Please enter your query: ")
message_history.append({'role':'user','content':user_query})
while True:
    response = client.chat.completions.parse(
        model="gpt-4o",
        response_format=Output,
        messages=message_history
    )
    raw_result=response.choices[0].message.content
    message_history.append({'role':'assistant','content':raw_result})
    parsed_result=response.choices[0].message.parsed
    if parsed_result.step == "START":
        print("I am initiating thinking mode to solve the question!")
        continue
    if parsed_result.step == "PLAN":
        print("I am doing ",parsed_result.content)
        continue
    if parsed_result.step == "TOOL":
        tool=parsed_result.tool
        input=parsed_result.input
        print("I am calling {tool}")
        re=available[tool](input)
        message_history.append({'role':'developer','content':json.dumps({'step':"OBSERVE",'tool':'tool','input':input,'output':re})})
        continue
    if parsed_result.step == "OUTPUT":
        print(parsed_result.content)
        break


