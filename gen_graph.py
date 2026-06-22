from src.control.agents.graphs.interview_graph import get_graph


def generate():
    g = get_graph()
    png_data = g.get_graph().draw_mermaid_png()
    with open("interview_graph_official.png", "wb") as f:
        f.write(png_data)
    print("Graph generated at interview_graph_official.png")


if __name__ == "__main__":
    generate()
