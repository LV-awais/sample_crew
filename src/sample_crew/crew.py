from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from .tools.custom_tool import CombinedTool, SerperSearchTool
from crewai import LLM  # Ensure this is imported if you’re using LLM

load_dotenv()

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class SampleCrew:
    """SampleCrew crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    @agent
    def retrieve_suppliers(self) -> Agent:
        return Agent(
            config=self.agents_config['retrieve_suppliers'],
            tools=[SerperSearchTool()],
            verbose=True,
            allow_delegation=True,
        )

    @agent
    def domain_researcher_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['domain_researcher_agent'],
            tools=[CombinedTool(result_as_answer=True)],
            verbose=True,
        )

    @agent
    def ai_suppliers_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['ai_suppliers_writer'],
            verbose=True,
        )

    @task
    def retrieve_suppliers_task(self) -> Task:
        return Task(
            config=self.tasks_config['retrieve_suppliers_task'],
        )

    @task
    def domain_and_trustpilot_researcher_task(self) -> Task:
        return Task(
            config=self.tasks_config['domain_and_trustpilot_researcher_task'],
        )

    @task
    def ai_suppliers_write_task(self) -> Task:
        return Task(
            config=self.tasks_config['ai_suppliers_write_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AiSuppliers crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            planning=True,
            planning_llm=LLM(
                model="gemini/gemini-2.0-flash"
            )  # This line was incorrectly indented before
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
