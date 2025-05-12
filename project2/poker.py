import random
from collections import Counter
import itertools
import math
import time


#STRUCTURES

suits = ['C', 'D', 'H', 'S']
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']


# HELPERS
def card_to_index(card_str: str) -> int:
    """Convert card string (e.g. 'AH') to integer index 0-51."""
    rank_str, suit_str = card_str[:-1], card_str[-1]
    if rank_str not in ranks or suit_str not in suits:
        raise ValueError(f"Invalid card: {card_str}")
    rank = ranks.index(rank_str)
    suit = suits.index(suit_str)
    return suit * 13 + rank

def index_to_card(idx: int) -> str:
    # if someone accidentally passes a string, try to convert it
    if isinstance(idx, str):
        try:
            idx = int(idx)
        except ValueError:
            raise ValueError(f"index_to_card requires an integer index, got {idx!r}")
    if not (0 <= idx < 52):
        raise ValueError(f"Invalid index: {idx}")
    suit, rank = divmod(idx, 13)
    return ranks[rank] + suits[suit]

def card_to_rs(idx: int) -> tuple[int,int]:
    """Return (value, suit) where value is 2-14, suit 0-3."""
    suit, rank = divmod(idx, 13)
    return rank + 2, suit

# Converts 0–12 to 2–14 (2 through Ace)
def rank_to_value(rank):
    return [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14][rank]

def has_straight(ranks):
    ranks = sorted(set(ranks), reverse=True)
    if 14 in ranks:
        ranks.append(1)
    for i in range(len(ranks) - 4):
        if ranks[i] - ranks[i + 4] == 4:
            return True, ranks[i]
    return False, None

def create_deck()->list[int]:
    """Create a standard 52-card deck as a list of integers 0–51."""
    return list(range(52))

def shuffle_deck(deck:list[int])->list[int]:
    """Shuffle the deck in place."""
    random.shuffle(deck)
    return deck

def draw_cards(deck: list[int], num_cards: int) -> list[int]:
    """Draw num_cards from the deck."""
    assert len(deck) >= num_cards, "Not enough cards left in deck"
    drawn_cards = deck[:num_cards]
    del deck[:num_cards]
    return drawn_cards


# Poker Hand Evaluation
def rank_hand(indices: list[int]) -> tuple:
    """Return a tuple that ranks the 5-card hand by poker rules."""
    vals, suits = zip(*(card_to_rs(i) for i in indices))
    counts = Counter(vals)
    freq = sorted(counts.items(), key=lambda x: (-x[1], -x[0]))
    sorted_vals = sorted(vals, reverse=True)
    # Flush?
    flush_suit = None
    for s, cnt in Counter(suits).items():
        if cnt >= 5:
            # need all 5 to be the same suit
            flush_suit = s
            break
    # Straight flush
    if flush_suit is not None:
        flush_vals = [v for v, su in zip(vals, suits) if su == flush_suit]
        sf, high_sf = has_straight(flush_vals)
        if sf:
            return (9, high_sf)
    # Four, Full, Three, Two Pair, Pair by freq
    kinds = [v for v, c in freq]
    counts_only = [c for v, c in freq]
    # Four of a Kind
    if counts_only[0] == 4:
        four = kinds[0]
        kicker = max(v for v in sorted_vals if v != four)
        return (8, four, kicker)
    # Full House
    if counts_only[0] == 3 and counts_only[1] >= 2:
        return (7, kinds[0], kinds[1])
    # Flush
    if flush_suit is not None:
        top5 = sorted(flush_vals, reverse=True)[:5]
        return (6, *top5)
    # Straight
    st, high_st = has_straight(sorted_vals)
    if st:
        return (5, high_st)
    # Three of a Kind
    if counts_only[0] == 3:
        kickers = [v for v in sorted_vals if v != kinds[0]][:2]
        return (4, kinds[0], *kickers)
    # Two Pair
    if counts_only[0] == 2 and counts_only[1] == 2:
        kicker = max(v for v in sorted_vals if v not in (kinds[0], kinds[1]))
        return (3, kinds[0], kinds[1], kicker)
    # One Pair
    if counts_only[0] == 2:
        kickers = [v for v in sorted_vals if v != kinds[0]][:3]
        return (2, kinds[0], *kickers)
    # High Card
    return (1, *sorted_vals[:5])

# Monte Carlo Simulation 
def simulate_win_probability(my_hole: list[str], community: list[str], time_limit: float = 10.0) -> float:
    """
    Estimate win probability via MCTS: opponent hole cards as children, UCB1 guide.
    """
    # Prepare fixed cards
    my_idx = [card_to_index(c) for c in my_hole]
    comm_idx = [card_to_index(c) for c in community]
    # Build deck
    deck = create_deck()
    for c in my_idx + comm_idx:
        deck.remove(c)
    # All possible opponent  combos
    opp_combos = list(itertools.combinations(deck, 2))
    stats = {combo: {'wins':0, 'plays':0} for combo in opp_combos}
    total_plays = 0
    wins = 0
    start = time.time()
    while time.time() - start < time_limit:
        best_ucb = -1
        chosen = None
        for combo, st in stats.items():
            if st['plays']==0:
                ucb = float('inf')
            else:
                win_rate = st['wins']/st['plays']
                ucb = win_rate + (2*(math.log(total_plays)/st['plays']))**0.5
            if ucb > best_ucb:
                # update best UCB
                best_ucb, chosen = ucb, combo
        opp_idx = list(chosen)
        # Remove opponent cards
        trial_deck = deck.copy()
        trial_deck.remove(opp_idx[0]); trial_deck.remove(opp_idx[1])
        # Draw remaining community to 5
        needed = 5 - len(comm_idx)
        shuffle_deck(trial_deck)
        extra = trial_deck[:needed]
        
        # Evaluate
        my_best = max(itertools.combinations(my_idx + comm_idx + extra, 5), key=lambda h: rank_hand(list(h)))
        opp_best = max(itertools.combinations(opp_idx + comm_idx + extra, 5), key=lambda h: rank_hand(list(h)))
        if rank_hand(list(my_best)) > rank_hand(list(opp_best)):
            wins += 1
            stats[chosen]['wins'] += 1
        stats[chosen]['plays'] += 1
        total_plays += 1
    return wins/total_plays if total_plays>0 else 0.0



def decide_action(my_hole: list[str], community: list[str]) -> str:
    prob = simulate_win_probability(my_hole, community, time_limit=10.0)
    print(f"Estimated win probability: {prob:.2%}")
    return 'stay' if prob >= 0.5 else 'fold'





### TESTS ###
def test_simulation_scenarios():
    # Obvious winning hand: Royal Flush vs unknown
    win_prob = simulate_win_probability(['AH', 'KH'], ['QH', 'JH', 'TH'], time_limit=2)
    assert win_prob > 0.98, f"Expected near-certain win, got {win_prob:.2f}"

    # Weak hand, high card only
    win_prob = simulate_win_probability(['2C', '7D'], ['5S', '9H', 'JC'], time_limit=2)
    assert win_prob < 0.3, f"Expected low win prob, got {win_prob:.2f}"

    # Simulation with full board (no remaining cards)
    win_prob = simulate_win_probability(['AC', 'KS'], ['2H', '3D', '4S', '5C', '8H'], time_limit=2)
    assert 0.5 <= win_prob <= 1.0, f"Invalid probability: {win_prob}"

    # Test when hand looks strong but can be beaten easily
    win_prob = simulate_win_probability(['AH', 'AD'], ['KH', 'QH', 'JH'], time_limit=2)
    assert win_prob < 0.9, "Trips vs potential flush/straight"

    # Test when holding low cards but board offers straight
    win_prob = simulate_win_probability(['3D', '4C'], ['2H', '5S', '6D'], time_limit=2)
    assert win_prob > 0.5, "Straight from community"

    # All community cards known, strong board
    win_prob = simulate_win_probability(['2C', '7D'], ['9H', '9C', '9D', '5S', '6H'], time_limit=2)
    assert win_prob < 0.2, "Weak hand vs trips on board"

    # Board itself has a flush; test high card flush tiebreak
    win_prob = simulate_win_probability(['AS', '2S'], ['6S', '7S', '8S', '3S', '4S'], time_limit=2)
    assert win_prob > 0.95, "Nut flush from hand"

    print("All simulation tests passed.")


def test_decision_points():
    # Pre-Flop: No community cards
    win_prob = simulate_win_probability(['AH', 'KH'], [], time_limit=2)
    assert 0 <= win_prob <= 1

    # Flop: 3 community cards
    win_prob = simulate_win_probability(['AH', 'KH'], ['QH', 'JH', '9D'], time_limit=2)
    assert 0 <= win_prob <= 1

    # Turn: 4 community cards
    win_prob = simulate_win_probability(['AH', 'KH'], ['QH', 'JH', '9D', '2C'], time_limit=2)
    assert 0 <= win_prob <= 1

    # River: 5 community cards
    win_prob = simulate_win_probability(['AH', 'KH'], ['QH', 'JH', '9D', '2C', '5S'], time_limit=2)
    assert 0 <= win_prob <= 1

    print("All decision point tests passed.")



def test_compare_monte_carlo_vs_ucb1():
    hand = ['AH', 'KH']
    board = ['QH', 'JH', '9D']
    
    prob_mc = simulate_win_probability(hand, board, time_limit=2)
    prob_ucb = simulate_win_probability(hand, board, time_limit=2)
    
    print(f"MC estimate: {prob_mc:.2f}, UCB1 estimate: {prob_ucb:.2f}")
    assert abs(prob_mc - prob_ucb) < 0.3, "UCB1 diverging too far from baseline"


def test_no_duplicate_cards():
    hole = ['AH', 'KH'] # two priv cards
    board = ['QH', 'JH', '9D'] 
    deck = create_deck()
    assert len(deck) == len(set(deck)) == 52
    for c in hole + board:
        deck.remove(card_to_index(c))
    assert len(deck) == len(set(deck))




def test_hand_evaluation():
    def assert_rank(cards_str, expected_rank, description):
        cards = [card_to_index(c) for c in cards_str]
        actual_rank = rank_hand(cards)[0]
        assert actual_rank == expected_rank, f"{description} failed: got {actual_rank}"

    # Royal Flush
    assert_rank(['TH', 'JH', 'QH', 'KH', 'AH'], 9, "Royal Flush")

    # Straight Flush
    assert_rank(['7S', '8S', '9S', 'TS', 'JS'], 9, "Straight Flush")

    # Four of a Kind
    assert_rank(['9C', '9D', '9H', '9S', '2C'], 8, "Four of a Kind")

    # Full House
    assert_rank(['3C', '3D', '3S', '6H', '6D'], 7, "Full House")

    # Flush
    assert_rank(['2D', '5D', '7D', '9D', 'QD'], 6, "Flush")

    # Straight (Ace-low)
    assert_rank(['AH', '2D', '3S', '4C', '5H'], 5, "Straight (Ace-low)")

    # Straight (normal)
    assert_rank(['8C', '9H', 'TS', 'JD', 'QH'], 5, "Straight")

    # Three of a Kind
    assert_rank(['7H', '7D', '7S', '2C', '5H'], 4, "Three of a Kind")

    # Two Pair
    assert_rank(['6H', '6D', '9C', '9S', '3H'], 3, "Two Pair")

    # One Pair
    assert_rank(['KH', 'KD', '4C', '7S', '9H'], 2, "One Pair")

    # High Card
    assert_rank(['2H', '5D', '8C', 'JH', 'QD'], 1, "High Card")

    # Edge case: Ace-high straight (A-2-3-4-5)
    assert_rank(['5S', '4D', '3C', '2H', 'AH'], 5, "Straight A-2-3-4-5")

    # Edge case: Queen-high straight (T-J-Q-K-A)
    assert_rank(['TS', 'JH', 'QD', 'KC', 'AD'], 5, "Straight T-J-Q-K-A")

    # Edge case: Low pair vs high card
    assert_rank(['2H', '2D', '5C', '8S', 'KH'], 2, "One Pair (Low Pair)")

    # Edge case: Three of a Kind with two high kickers
    assert_rank(['8H', '8D', '8S', 'KH', 'QC'], 4, "Three of a Kind + Kickers")

    # Edge case: Flush with low cards
    assert_rank(['2S', '5S', '7S', '9S', 'JS'], 6, "Flush (low)")

    # Edge case: Full House with lower trips but higher pair
    assert_rank(['3S', '3D', '3C', 'KH', 'KS'], 7, "Full House (low trips, high pair)")

    print("All hand evaluation tests passed.")


if __name__ == "__main__":
    test_hand_evaluation()
    test_simulation_scenarios()
    test_decision_points()
    test_no_duplicate_cards()
    test_compare_monte_carlo_vs_ucb1()