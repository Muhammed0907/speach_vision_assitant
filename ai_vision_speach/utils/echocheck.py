import difflib

def is_likely_system_echo(user_text, last_response):
    similarity_threshold = 0.6  # Similarity threshold to identify system echo (0-1, higher means more similar)

    """
    Check if the user's text is likely to be the system hearing itself.
    
    Args:
        user_text: The text heard from the user
        last_response: The last response given by the assistant
        
    Returns:
        bool: True if it seems like an echo, False otherwise
    """
    
    # If we have no last response, it can't be an echo
    if not last_response or not user_text:
        return False
        
    # Convert both to lowercase for better comparison
    user_text_lower = user_text.lower()
    last_response_lower = last_response.lower()
    
    # Method 1: Simple containment check (if user text is almost completely inside assistant's last response)
    # Find the longest common substring
    matcher = difflib.SequenceMatcher(None, user_text_lower, last_response_lower)
    match = matcher.find_longest_match(0, len(user_text_lower), 0, len(last_response_lower))
    
    # If the match covers most of the user text, it's likely an echo
    longest_match_ratio = match.size / len(user_text_lower) if len(user_text_lower) > 0 else 0
    
    # Method 2: Similarity ratio between the two strings
    similarity_ratio = matcher.ratio()
    
    # Log for debugging
    print(f"Echo check - Match ratio: {longest_match_ratio:.2f}, Similarity: {similarity_ratio:.2f}")
    
    # Determine if it's an echo based on either method
    is_echo = (longest_match_ratio > 0.7) or (similarity_ratio > similarity_threshold)
    
    if is_echo:
        print(f"Detected system echo: '{user_text}' is similar to assistant's response")
        print(f"Last assistant response: '{last_response}'")
    
    return is_echo

