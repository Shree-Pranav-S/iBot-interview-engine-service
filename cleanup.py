import glob
import re


def main():
    files = glob.glob("src/control/agents/nodes/*.py")
    for f in files:
        with open(f) as file:
            content = file.read()

        content = re.sub(
            r"^\s*\"(skip_insisted|nudge_given|skip_requested|silence_attempt|awaiting_think_decision|think_timer_active|thinking_expires_at)\":.*?\n",
            "",
            content,
            flags=re.MULTILINE,
        )

        with open(f, "w") as file:
            file.write(content)


if __name__ == "__main__":
    main()
