from crewai import LLM, Agent, Task, Crew, Process
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = LLM(
    model="gemini/gemini-3.1-flash-lite",
    api_key=api_key,
    temperature=0.7
)
#Reasearch agent
research_agent = Agent(
    role="Movie Researcher",
    goal="Research popular {genre} telugu movies and their story lines",
    backstory="You are an experienced movie researcher",
    llm=llm,
    verbose=True
)

#movie name generator
name_agent = Agent(
    role="Movie name Generator",
    goal="Create creative and unique {genre} movie names",
    backstory="You are an expert movie title creator",
    llm=llm,
    verbose=True
)

#subtitle agent
subtitle_agent = Agent(
    role="Subtitle Generator",
    goal="Creative attractive subtitles for movie titles",
    backstory="You are an expert movie tagline writer",
    llm=llm,
    verbose=True
)

research_task = Task(
    description="""
    Research 5 popular {genre} telugu movie.
    Collect their names, release years, and short storylines
    """,
    expected_output="Five movies with names, release years and storylines",
    agent=research_agent
)

name_task = Task(
    description="""
    Based on the research results,
    generate 5 unique and creative {genre} movie titles.
    """,
    expected_output="Five creative movie titles.",
    agent=name_agent,
    context=[research_task]
)
subtitle_task = Task(
    description="""
    Create a catchy subtitle or tagline
    for each movie title generated in the previous task.
    Make sure the subtitles match the {genre} genre.
    """,
    expected_output="Five movie titles with their corresponding subtitles.",
    agent=subtitle_agent,
    context=[research_task, name_task]
)

movie_crew = Crew(
    agents=[
        research_agent,
        name_agent,
        subtitle_agent
    ],
    tasks=[
        research_task,
        name_task,
        subtitle_task
    ],
    process=Process.sequential,
    verbose=True
)

genre = input("Enter your movie genre: ")

result = movie_crew.kickoff(
    inputs={"genre": genre}
)
print("Result")
print(result)


# uv run main.py