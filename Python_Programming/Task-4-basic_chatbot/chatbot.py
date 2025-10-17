# chatbot.py
import random
import time


class BasicChatbot:
    def __init__(self):
        self.name = "CodeAlpha Bot"
        self.creator = "Your Name"
        self.responses = self.initialize_responses()
        self.default_responses = [
            "That's interesting! Tell me more.",
            "I see. Could you elaborate?",
            "Why do you say that?",
            "How does that make you feel?",
            "That's quite fascinating!",
            "I'm learning from our conversation!",
            "Let's talk about something else. What are your hobbies?"
        ]

    def initialize_responses(self):
        """Initialize all predefined responses"""
        return {
            # Greetings
            "hello": ["Hello! 👋", "Hi there!", "Hey! Nice to meet you!"],
            "hi": ["Hi! How can I help you?", "Hello! What's on your mind?"],
            "hey": ["Hey there! 😊", "Hey! How's your day going?"],

            # How are you
            "how are you": ["I'm functioning well, thank you!", "I'm great! How about you?",
                            "All systems operational! 😄"],
            "how are you doing": ["I'm doing fantastic!", "Pretty good for a chatbot! How about you?"],

            # Goodbyes
            "bye": ["Goodbye! 👋", "See you later!", "Bye! Have a great day!"],
            "goodbye": ["Farewell! 👋", "Take care!", "Until next time!"],
            "exit": ["Thanks for chatting! 👋", "It was nice talking to you!", "Chat with you again soon!"],

            # Thanks
            "thanks": ["You're welcome! 😊", "Anytime!", "Happy to help!"],
            "thank you": ["My pleasure! 🌟", "No problem at all!", "You're very welcome!"],

            # Name
            "what is your name": [f"I'm {self.name}! Nice to meet you!", f"You can call me {self.name}.",
                                  f"I'm {self.name}, your friendly chatbot!"],
            "who are you": [f"I'm {self.name}, a simple rule-based chatbot.",
                            f"I'm {self.name} created for CodeAlpha internship!"],

            # Creator
            "who made you": [f"I was created by {self.creator} for the CodeAlpha Python Internship!",
                             f"My developer is {self.creator}!"],
            "who created you": [f"I'm a project by {self.creator} for learning Python!",
                                f"{self.creator} built me as part of their internship!"],

            # Help
            "help": ["I can chat about simple topics! Try saying hello, asking how I am, or telling me about your day.",
                     "I'm a basic chatbot. You can greet me, ask about my name, or just have a simple conversation!"],

            # Feelings
            "i'm sad": ["I'm sorry to hear that. 😔 Remember, every cloud has a silver lining!",
                        "I'm here for you. Things will get better! 💫"],
            "i'm happy": ["That's wonderful! 😊 Spread the happiness around!", "Great to hear that! Keep smiling! 🌟"],

            # Time
            "what time is it": [f"The current time is {time.strftime('%H:%M:%S')}.",
                                f"It's {time.strftime('%I:%M %p')} right now."],
            "what day is it": [f"Today is {time.strftime('%A, %B %d, %Y')}.", f"It's {time.strftime('%A')} today!"],

            # Weather (placeholder)
            "how is the weather": ["I don't have real-time weather data, but I hope it's nice where you are! ☀️",
                                   "I'm just a simple bot, but I hope the weather is pleasant for you!"],

            # Hobbies
            "what do you like": ["I enjoy chatting with people like you! 💬", "I like learning from our conversations!",
                                 "My favorite hobby is helping users!"],

            # CodeAlpha
            "codealpha": ["CodeAlpha is an amazing platform for internships! 🚀",
                          "I was created as part of the CodeAlpha Python Internship program!",
                          "CodeAlpha helps students learn real-world programming skills!"],

            # Python
            "python": ["Python is a wonderful programming language! 🐍",
                       "I'm built with Python - it's great for beginners and experts alike!",
                       "Python makes programming fun and easy!"]
        }

    def get_response(self, user_input):
        """Get appropriate response based on user input"""
        user_input = user_input.lower().strip()

        # Check for exact matches first
        for pattern, responses in self.responses.items():
            if pattern in user_input:
                return random.choice(responses)

        # Check for partial matches with some flexibility
        if any(word in user_input for word in ["sad", "unhappy", "depressed"]):
            return random.choice(self.responses["i'm sad"])

        if any(word in user_input for word in ["happy", "joy", "excited"]):
            return random.choice(self.responses["i'm happy"])

        if any(word in user_input for word in ["time", "clock"]):
            return random.choice(self.responses["what time is it"])

        if any(word in user_input for word in ["day", "date", "today"]):
            return random.choice(self.responses["what day is it"])

        if any(word in user_input for word in ["weather", "rain", "sunny"]):
            return random.choice(self.responses["how is the weather"])

        if any(word in user_input for word in ["hobby", "like", "interest"]):
            return random.choice(self.responses["what do you like"])

        # Default response if no match found
        return random.choice(self.default_responses)

    def display_welcome(self):
        """Display welcome message"""
        print("\n" + "=" * 60)
        print("🤖 WELCOME TO BASIC CHATBOT!")
        print("=" * 60)
        print(f"I'm {self.name}, your friendly rule-based chatbot.")
        print("I can respond to greetings, simple questions, and have basic conversations.")
        print("Type 'bye', 'exit', or 'quit' to end our chat.")
        print("Type 'help' to see what I can do.")
        print("=" * 60)

    def start_chat(self):
        """Start the chatbot conversation"""
        self.display_welcome()

        print(f"\n{self.name}: Hello! How can I assist you today? 😊")

        while True:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Check for exit conditions
            if user_input.lower() in ['bye', 'goodbye', 'exit', 'quit']:
                print(f"\n{self.name}: {random.choice(self.responses['bye'])}")
                break

            # Check for empty input
            if not user_input:
                print(f"\n{self.name}: I didn't catch that. Could you please type something?")
                continue

            # Get and display response
            response = self.get_response(user_input)

            # Simulate "typing" effect for more natural conversation
            print(f"\n{self.name}: ", end="", flush=True)
            time.sleep(0.5)  # Small delay for realism
            print(response)

            # Special case: if user asks for help, show capabilities
            if "help" in user_input.lower():
                self.show_capabilities()

    def show_capabilities(self):
        """Show what the chatbot can do"""
        print("\n" + "-" * 50)
        print("💡 THINGS I CAN RESPOND TO:")
        print("-" * 50)
        categories = {
            "Greetings": ["hello", "hi", "hey"],
            "Feelings": ["how are you", "i'm sad", "i'm happy"],
            "Information": ["what is your name", "who are you", "who made you"],
            "Time/Date": ["what time is it", "what day is it"],
            "Goodbyes": ["bye", "goodbye", "exit"]
        }

        for category, examples in categories.items():
            print(f"• {category}: {', '.join(examples)}")
        print("-" * 50)


def main():
    """Main function to run the chatbot"""
    chatbot = BasicChatbot()

    print("🚀 BASIC CHATBOT - CodeAlpha Internship Task 4")
    print("==============================================")

    try:
        chatbot.start_chat()
    except KeyboardInterrupt:
        print(f"\n\n{chatbot.name}: Oops! Looks like you interrupted me. Goodbye! 👋")
    except Exception as e:
        print(f"\n\n{chatbot.name}: Something went wrong! Error: {e}")

    print("\n👋 Thanks for chatting with me!")


# Run the chatbot if this file is executed directly
if __name__ == "__main__":
    main()