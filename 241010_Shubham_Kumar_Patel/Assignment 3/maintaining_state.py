import asyncio
from llama_index.utils.workflow import draw_all_possible_flows

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context
)

class SetupEvent(Event):
    query:str

class StepTwoEvent(Event):
    query:str

class StatefulFlow(Workflow):
    @step
    async def start(self, ctx: Context, ev:StartEvent) -> SetupEvent| StepTwoEvent:
        db = await ctx.get("some_database", default=None)
        if db is None:
            print("Need to load data")
            return SetupEvent(query=ev.query)
        else:
            print("Data already loaded")
            return StepTwoEvent(query=ev.query)
    
    @step
    async def setup(self, ctx:Context, ev:SetupEvent) -> StartEvent:
        #load data
        await ctx.set("some_database",[1,2,3])
        return StartEvent(query=ev.query)
    @step
    async def step_two(self, ctx: Context, ev: StepTwoEvent) -> StopEvent:
        # do something with the data
        print("Data is ", await ctx.get("some_database"))

        return StopEvent(result=await ctx.get("some_database"))

async def main():
    w = StatefulFlow(timeout=10, verbose=False)
    result = await w.run(query="Some query")
    print(result)
    draw_all_possible_flows(
    StatefulFlow,
    filename="basic_workflow.html",
    # Optional, can limit long event names in your workflow
    # Can help with readability
    # max_label_length=10,
    )

if __name__=="__main__":
    asyncio.run(main())