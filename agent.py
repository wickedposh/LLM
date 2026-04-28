from openai import OpenAI
from dotenv import load_dotenv
import requests,json
load_dotenv()


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
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type":"json_object"},
        messages=message_history
    )
    raw_result=response.choices[0].message.content
    message_history.append({'role':'assistant','content':raw_result})
    parsed_result=json.loads(raw_result)
    if parsed_result.get('step') == "START":
        print("I am initiating thinking mode to solve the question!")
        continue
    if parsed_result.get('step') == "PLAN":
        print("I am doing ",parsed_result.get("content"))
        continue
    if parsed_result.get('step') == "TOOL":
        tool=parsed_result.get("tool")
        input=parsed_result.get("input")
        print("I am calling {tool}")
        re=available[tool](input)
        message_history.append({'role':'developer','content':json.dumps({'step':"OBSERVE",'tool':'tool','input':input,'output':re})})
        continue
    if parsed_result.get('step') == "OUTPUT":
        print(parsed_result.get("content"))
        break


