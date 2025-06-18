from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
)
from llama_index.utils.workflow import draw_all_possible_flows
import random


class FirstEvent(Event):
    first_output: str


class SecondEvent(Event):
    second_output: str


class LoopEvent(Event):
    loop_output: str


class MyWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> FirstEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("Bad thing happened")
            return LoopEvent(loop_output="Back to step one.")
        else:
            print("Good thing happened")
            return FirstEvent(first_output="First step complete.")

    @step
    async def step_two(self, ev: FirstEvent) -> SecondEvent:
        print(ev.first_output)
        return SecondEvent(second_output="Second step complete.")

    @step
    async def step_three(self, ev: SecondEvent) -> StopEvent:
        print(ev.second_output)
        return StopEvent(result="Workflow complete.")


w = MyWorkflow(timeout=10, verbose=False)


async def main():
    result = await w.run(first_input="Start the workflow.")
    print(result)
    draw_all_possible_flows(MyWorkflow, filename="multi_step_workflow.html")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
