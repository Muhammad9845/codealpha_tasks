# hangman_game.py
import random


class HangmanGame:
    def __init__(self):
        # Predefined list of 5 words as per requirements
        self.words = ["PYTHON", "PROGRAMMING", "COMPUTER", "ALGORITHM", "DEVELOPER"]
        self.max_attempts = 6
        self.reset_game()

    def reset_game(self):
        """Reset the game state for a new game"""
        self.secret_word = random.choice(self.words)
        self.guessed_letters = set()
        self.correct_letters = set()
        self.incorrect_guesses = 0
        self.game_over = False
        self.won = False

    def display_word(self):
        """Display the word with guessed letters revealed"""
        display = ""
        for letter in self.secret_word:
            if letter in self.correct_letters:
                display += letter + " "
            else:
                display += "_ "
        return display.strip()

    def display_hangman(self):
        """Display hangman ASCII art based on incorrect guesses"""
        stages = [
            """
               ------
               |    |
               |
               |
               |
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |
               |
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |    |
               |
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |   /|
               |
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |   /|\\
               |
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |   /|\\
               |   /
               |
            --------
            """,
            """
               ------
               |    |
               |    O
               |   /|\\
               |   / \\
               |
            --------
            """
        ]
        return stages[self.incorrect_guesses]

    def get_guess(self):
        """Get and validate user input"""
        while True:
            guess = input("\nEnter a letter: ").upper().strip()

            if len(guess) != 1:
                print("❌ Please enter exactly one letter!")
                continue

            if not guess.isalpha():
                print("❌ Please enter a valid letter (A-Z)!")
                continue

            if guess in self.guessed_letters:
                print("❌ You already guessed that letter!")
                continue

            return guess

    def process_guess(self, guess):
        """Process the user's guess"""
        self.guessed_letters.add(guess)

        if guess in self.secret_word:
            self.correct_letters.add(guess)
            print(f"✅ Good guess! '{guess}' is in the word!")

            # Check if player won
            if all(letter in self.correct_letters for letter in self.secret_word):
                self.game_over = True
                self.won = True
        else:
            self.incorrect_guesses += 1
            print(f"❌ Sorry, '{guess}' is not in the word!")

            # Check if player lost
            if self.incorrect_guesses >= self.max_attempts:
                self.game_over = True
                self.won = False

    def display_game_status(self):
        """Display current game status"""
        print("\n" + "=" * 50)
        print(self.display_hangman())
        print(f"\nWord: {self.display_word()}")
        print(f"Incorrect guesses: {self.incorrect_guesses}/{self.max_attempts}")
        print(f"Guessed letters: {', '.join(sorted(self.guessed_letters)) if self.guessed_letters else 'None'}")
        print("=" * 50)

    def play_round(self):
        """Play one round of Hangman"""
        print("🎮 WELCOME TO HANGMAN GAME!")
        print("=" * 50)
        print("Rules:")
        print("- Guess the word one letter at a time")
        print(f"- You have {self.max_attempts} incorrect guesses allowed")
        print("- Only letters A-Z are allowed")
        print("=" * 50)

        while not self.game_over:
            self.display_game_status()
            guess = self.get_guess()
            self.process_guess(guess)

        # Game over - display result
        self.display_game_status()

        if self.won:
            print(f"\n🎉 CONGRATULATIONS! You won!")
            print(f"🏆 The word was: {self.secret_word}")
        else:
            print(f"\n💀 GAME OVER! You've been hanged!")
            print(f"📖 The word was: {self.secret_word}")

    def play_again(self):
        """Ask if player wants to play again"""
        while True:
            choice = input("\nWould you like to play again? (Y/N): ").upper().strip()
            if choice in ['Y', 'YES']:
                return True
            elif choice in ['N', 'NO']:
                return False
            else:
                print("❌ Please enter 'Y' for Yes or 'N' for No")


def main():
    """Main function to run the Hangman game"""
    game = HangmanGame()

    print("🚀 HANGMAN GAME - CodeAlpha Internship Task")
    print("============================================")

    while True:
        game.reset_game()
        game.play_round()

        if not game.play_again():
            print("\n👋 Thanks for playing Hangman! Goodbye!")
            break


# Run the game if this file is executed directly
if __name__ == "__main__":
    main()