
from pydantic import BaseModel, ConfigDict,Field, model_validator,computed_field,field_validator
from typing import List,Optional
from datetime import datetime
mealtype=["breakfast","lunch","dinner","snack"]
class Dish():
    def __init__(self,name,sugar=0,calories=0,protein=0,im=None):
        self.name=name
        self.sugar=sugar
        self.calories=calories
        self.protein=protein
        self.image=im


class Meal(BaseModel):
        dishes=List[Dish]
        date=datetime
        model_config=ConfigDict(
            json_encoders={datetime: lambda v: v.strftime("%Y-%m-%d")}
        )## to get datetime string when we turn it into dictionary
        type=str
        @field_validator('type',mode='before')
        def type_validator(v):
            if v in mealtype:
                return v
            else:
                raise ValueError("wrong meal type! please choose among breakfast,lunch,dinner,snack")
        _time=str
        @field_validator('_time')
        def conversion(v):
            if int(v[-2:])>=60:
                raise ValueError("minute should be less than or equal to 60")
            t=int("".join(char for char in v if v.isdigit()))
            if 0000<=t<=2400:
                return t
            else:
                raise ValueError("time must be in 00:00 24-hr clock formats")
        @computed_field
        @property
        def sugar(self)->float:
            ss = (b.sugar for b in self.dishes)
            return sum(ss)
        @property
        def kcal(self)->float:
            kk= (b.calories for b in self.dishes)
            return sum(kk)
        @property
        def protein(self)->float:
            pp= (b.protein for b in self.dishes)
            return sum(pp)




class Exercise():
    def __init__(self,name,duration,calories):
        self.name=name
        self.duration=duration
        self.calories=calories
    @classmethod
    def from_pdrow(cls,data):
        return cls(data["name"],data["duration"],data["calories"])

class Weight(BaseModel):
    date:str
    weight:float








class User(BaseModel):
    id:int
    email:str=Field(regex=r'[\w\.-]+@[\w\.-]+\.\w+', description="email")
    password:str=Field(min_length=6,description="password")
    name:str
    is_active:bool
user_data={
    "id":1,
    "email":"dsajsdlfk@naver.com",
    "password":"merong", }
user=User(**user_data)
class Signup(BaseModel):
    password:str
    confirm:str
    @model_validator(mode='after')
    def mathc(selfcls,values):
        if values.password != values.confirm:
            raise ValueError("password and confirmation of it must be equal")
        return values
class Comment(BaseModel):
    id:int
    comment:str
    replies:Optional[List['Comment']]
Comment.model_rebuild()

c=Comment(
    id=1,
    comment="nononono",
    replies=[
        Comment(id=2,comment="nonseono2"),
        Comment(id=3,comment="nonseono3",replies=[
            Comment(id=4,comment="nonseono4")])
    ]
)