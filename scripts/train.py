from src.rl import DQNTrainer, TrainerConfig


def main() -> None:
    trainer = DQNTrainer(TrainerConfig())
    logs = trainer.train(episodes=200)
    print("Training done.")
    print("Last log:", logs[-1])


if __name__ == "__main__":
    main()
