from .stages.initialize_stage import InitializeStage


def main() -> None:
    detector = InitializeStage()
    detector.run()


if __name__ == "__main__":
    main()
