import os
import sys
import warnings
import logging
from contextlib import contextmanager

# Store the original stderr and redirect it to /dev/null to suppress OGR warnings
# This effectively silences all error messages during module imports
original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import argparse
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_collection import collect_all_data
from src.data_processing import process_and_integrate_data
from src.agent import UrbanDataAgent
import src.config as config

def show_banner():
    """Displays the welcome banner with application description
    
    Creates a bordered text box containing the application name and
    a brief description in Romanian.
    """
    print("\n" + "=" * 80)
    print("""
    🌳 ASISTENT PENTRU DATE URBANE BUCUREȘTI 🏙️
    
    Un instrument interactiv de analiză geospațială pentru explorarea
    datelor urbane din București folosind întrebări în limbaj natural.
    """)
    print("=" * 80 + "\n")

def show_help():
    """Displays help information and example queries
    
    Shows a list of example questions that users can ask the system
    and instructions for exiting or getting help.
    """
    print("\nPoți adresa întrebări despre mediul urban din București, precum:")
    print("  - Care sector are cele mai multe parcuri?")
    print("  - Care sunt parcurile cu cea mai bună calitate a aerului?")
    print("  - Găsește un traseu între Parcul Herăstrău și Grădina Cișmigiu")
    print("  - Unde ar fi bune locații pentru parcuri noi?")
    print("  - Creează un grafic comparând sectoarele după numărul de parcuri")
    print("  - Arată-mi harta de căldură pentru calitatea aerului")
    print("  - Caută parcuri care conțin 'Tineretului'")
    print("\nScrie 'ieșire', 'quit' sau 'bye' pentru a încheia conversația.")
    print("Scrie 'ajutor' pentru a vedea din nou aceste informații.\n")

def parse_arguments():
    """Processes command line arguments for the CLI application
    
    Sets up argument parsing for force-update and verbose logging options
    that control application behavior.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Asistent Date Urbane București CLI")
    parser.add_argument(
        "-f", "--force-update",
        action="store_true",
        help="Forțează actualizarea tuturor surselor de date, ignorând cache-ul"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Activează logarea detaliată"
    )
    return parser.parse_args()

def main():
    """Main entry point for the CLI application
    
    Controls the entire application flow:
    1. Parses command-line arguments
    2. Displays welcome information
    3. Loads and processes urban data
    4. Initializes the conversational agent
    5. Runs the interactive question-answer loop
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Display welcome banner
    show_banner()
    
    print("Inițializez Asistentul pentru Date Urbane București...")
    
    # 1. Collect all necessary urban data
    # This loads park information, air quality, traffic, and other datasets
    print("\n🔄  Colectare date urbane...")
    data_dict = collect_all_data(force_update=args.force_update)
    
    # Check if essential park and sector data was successfully loaded
    if not data_dict or 'parks_sectors' not in data_dict or data_dict['parks_sectors'].empty:
        print("❌  Eroare: Eșec în colectarea datelor esențiale. Verificați conexiunea la internet.")
        sys.exit(1)
    
    # 2. Process and integrate all collected data sources
    # This combines and harmonizes data from different sources
    print("\n🔄  Procesare și integrare date...")
    final_data = process_and_integrate_data(data_dict)
    
    # Ensure data processing succeeded
    if final_data is None:
        print("❌  Eroare: Procesarea datelor a eșuat. Nu se poate continua.")
        sys.exit(1)
    
    # 3. Initialize the conversational agent with the processed data
    # This prepares the NLP system to answer user questions
    print("\n🔄  Inițializare agent conversațional...")
    agent = UrbanDataAgent(final_data)
    
    # 4. Start interactive session with the user
    print("\n✅  Configurare completă! Începe sesiunea interactivă.")
    show_help()
    
    # Main interactive loop - continually process user questions
    while True:
        try:
            # Get user input
            user_input = input("\n👤 Tu: ")
            
            # Process special commands for exiting or help
            if user_input.lower() in ['ieșire', 'iesire', 'exit', 'quit', 'bye']:
                print("\nMulțumim că ați folosit Asistentul pentru Date Urbane București. La revedere!")
                break
            elif user_input.lower() in ['ajutor', 'help']:
                show_help()
                continue
            elif not user_input.strip():
                continue
                
            # Process the user's question through the agent
            print("\n🤔  Mă gândesc...")
            start_time = datetime.now()
            
            result = agent.process_query(user_input)
            response = result["response"] if isinstance(result, dict) else result
            visualization = result.get("visualization") if isinstance(result, dict) else None
            
            # Display the response with timing information
            elapsed_time = (datetime.now() - start_time).total_seconds()
            print(f"\n🤖  Asistent: {response}")
            if visualization:
                print(f"📊  Vizualizare: {visualization.lstrip('/')} (pornește aplicația web pentru a o vedea)")
            print(f"\n(Răspuns generat în {elapsed_time:.2f} secunde)")
            
        except KeyboardInterrupt:
            # Handle clean exit on Ctrl+C
            print("\n\nSesiune întreruptă. Se închide...")
            break
        except Exception as e:
            # Handle unexpected errors during query processing
            print(f"\n❌  A apărut o eroare neașteptată: {str(e)}")
            print("Vă rugăm să încercați din nou.")

if __name__ == "__main__":
    main()
