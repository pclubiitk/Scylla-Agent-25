import random
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
)
from llama_index.utils.workflow import draw_all_possible_flows
class BranchEvent1(Event):
    payload: str
    count: int = 0

class BranchEvent2(Event):
    payload: str
    count: int = 0


class LoopEvent1(Event):
    count: int = 0


class LoopEvent2(Event):
    count: int = 0
    
class CommonEvent(Event):
    payload : str


class BranchWorkflow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> BranchEvent1 | BranchEvent2:
        if random.randint(0, 1) == 0:
            print("Go to branch A")
            return BranchEvent1(payload="Branch A")
        else:
            print("Go to branch B")
            return BranchEvent2(payload="Branch B")

    @step
    async def step_one(self, ev: LoopEvent1 | BranchEvent1) -> LoopEvent1 | CommonEvent:
        count=ev.count
        print(f"Count: {count}")
        if count < 10:
            return LoopEvent1(count=count + 1)
        else:
            return CommonEvent(payload="Done")
        
    @step
    async def step_two(self, ev: LoopEvent2 | BranchEvent2) -> LoopEvent2 | CommonEvent:
        count = ev.count 
        print(f"Count: {count}")
        if count < 5:
            return LoopEvent1(count=count + 1)
        else:
            return CommonEvent(payload="Done")
    
    @step
    async def step_three(self,ev: CommonEvent) -> StopEvent:
        print(ev.payload)
        
async def main():
    w = BranchWorkflow(timeout=10, verbose=False)
    draw_all_possible_flows(BranchWorkflow, filename="workflow_graph.html")
    result = await w.run()
    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
