# prl_environment

## Vectorized observation
- A vectorized observation is encoded with a fixed length assuming 6 Players.
- Missing players will be filled with 0 bits.
- Observation is always encoded relative to observer, so that each bit has a fixed meaning
  * Note that the last acting player is always the furthest away, so it will thus be at position 5.
  * This is counter-intuitive for less than six players, but appropriate because the person left to the big blind is 
    * UTG position on a 6 player table and becomes MP for 5 players, CU for 4 players, and so on

- The offset to the acting player is one-hot encoded by the `player_acts_next` bit. 
- The button position is encoded by the `button_index` bit. 
- The dimensions are mostly one-hot encoded except for bet sizes and min_raise, which are normalized floats
- The vectorized observation has 565 dimensions

The dimensions are as returned from the 
[Steinberger PokerRL environment](https://github.com/hellovertex/steinberger)

`+` Our augmentation is appended to the end, that includes

- Player Hands with possible SEER mode that shows all hands
- Action history including actions for current hand, capped at 2 actions per stage (Preflop, Flop, Turn, River)




#### Example observation
                              ante:   0.0
                       small_blind:   0.02500000037252903
                         big_blind:   0.05000000074505806
                         min_raise:   0.15000000596046448
                           pot_amt:   0.0
                     total_to_call:   0.10000000149011612
                      p0_acts_next:   0.0
                      p1_acts_next:   1.0
                      p2_acts_next:   0.0
                      p3_acts_next:   0.0
                      p4_acts_next:   0.0
                      p5_acts_next:   0.0
                     round_preflop:   1.0
                        round_flop:   0.0
                        round_turn:   0.0
                       round_river:   0.0
                        side_pot_0:   0.0
                        side_pot_1:   0.0
                        side_pot_2:   0.0
                        side_pot_3:   0.0
                        side_pot_4:   0.0
                        side_pot_5:   0.0
                          stack_p0:   0.949999988079071
                       curr_bet_p0:   0.05000000074505806
        has_folded_this_episode_p0:   0.0
                       is_allin_p0:   0.0
             side_pot_rank_p0_is_0:   0.0
             side_pot_rank_p0_is_1:   0.0
             side_pot_rank_p0_is_2:   0.0
             side_pot_rank_p0_is_3:   0.0
             side_pot_rank_p0_is_4:   0.0
             side_pot_rank_p0_is_5:   0.0
                          stack_p1:   0.0
                       curr_bet_p1:   0.0
        has_folded_this_episode_p1:   0.0
                       is_allin_p1:   0.0
             side_pot_rank_p1_is_0:   0.0
             side_pot_rank_p1_is_1:   0.0
             side_pot_rank_p1_is_2:   0.0
             side_pot_rank_p1_is_3:   0.0
             side_pot_rank_p1_is_4:   0.0
             side_pot_rank_p1_is_5:   0.0
                          stack_p2:   0.0
                       curr_bet_p2:   0.0
        has_folded_this_episode_p2:   0.0
                       is_allin_p2:   0.0
             side_pot_rank_p2_is_0:   0.0
             side_pot_rank_p2_is_1:   0.0
             side_pot_rank_p2_is_2:   0.0
             side_pot_rank_p2_is_3:   0.0
             side_pot_rank_p2_is_4:   0.0
             side_pot_rank_p2_is_5:   0.0
                          stack_p3:   0.0
                       curr_bet_p3:   0.0
        has_folded_this_episode_p3:   0.0
                       is_allin_p3:   0.0
             side_pot_rank_p3_is_0:   0.0
             side_pot_rank_p3_is_1:   0.0
             side_pot_rank_p3_is_2:   0.0
             side_pot_rank_p3_is_3:   0.0
             side_pot_rank_p3_is_4:   0.0
             side_pot_rank_p3_is_5:   0.0
                          stack_p4:   0.0
                       curr_bet_p4:   0.0
        has_folded_this_episode_p4:   0.0
                       is_allin_p4:   0.0
             side_pot_rank_p4_is_0:   0.0
             side_pot_rank_p4_is_1:   0.0
             side_pot_rank_p4_is_2:   0.0
             side_pot_rank_p4_is_3:   0.0
             side_pot_rank_p4_is_4:   0.0
             side_pot_rank_p4_is_5:   0.0
                          stack_p5:   0.8999999761581421
                       curr_bet_p5:   0.10000000149011612
        has_folded_this_episode_p5:   0.0
                       is_allin_p5:   0.0
             side_pot_rank_p5_is_0:   0.0
             side_pot_rank_p5_is_1:   0.0
             side_pot_rank_p5_is_2:   0.0
             side_pot_rank_p5_is_3:   0.0
             side_pot_rank_p5_is_4:   0.0
             side_pot_rank_p5_is_5:   0.0
             0th_board_card_rank_0:   0.0
             0th_board_card_rank_1:   0.0
             0th_board_card_rank_2:   0.0
             0th_board_card_rank_3:   0.0
             0th_board_card_rank_4:   0.0
             0th_board_card_rank_5:   0.0
             0th_board_card_rank_6:   0.0
             0th_board_card_rank_7:   0.0
             0th_board_card_rank_8:   0.0
             0th_board_card_rank_9:   0.0
            0th_board_card_rank_10:   0.0
            0th_board_card_rank_11:   0.0
            0th_board_card_rank_12:   0.0
             0th_board_card_suit_0:   0.0
             0th_board_card_suit_1:   0.0
             0th_board_card_suit_2:   0.0
             0th_board_card_suit_3:   0.0
             1th_board_card_rank_0:   0.0
             1th_board_card_rank_1:   0.0
             1th_board_card_rank_2:   0.0
             1th_board_card_rank_3:   0.0
             1th_board_card_rank_4:   0.0
             1th_board_card_rank_5:   0.0
             1th_board_card_rank_6:   0.0
             1th_board_card_rank_7:   0.0
             1th_board_card_rank_8:   0.0
             1th_board_card_rank_9:   0.0
            1th_board_card_rank_10:   0.0
            1th_board_card_rank_11:   0.0
            1th_board_card_rank_12:   0.0
             1th_board_card_suit_0:   0.0
             1th_board_card_suit_1:   0.0
             1th_board_card_suit_2:   0.0
             1th_board_card_suit_3:   0.0
             2th_board_card_rank_0:   0.0
             2th_board_card_rank_1:   0.0
             2th_board_card_rank_2:   0.0
             2th_board_card_rank_3:   0.0
             2th_board_card_rank_4:   0.0
             2th_board_card_rank_5:   0.0
             2th_board_card_rank_6:   0.0
             2th_board_card_rank_7:   0.0
             2th_board_card_rank_8:   0.0
             2th_board_card_rank_9:   0.0
            2th_board_card_rank_10:   0.0
            2th_board_card_rank_11:   0.0
            2th_board_card_rank_12:   0.0
             2th_board_card_suit_0:   0.0
             2th_board_card_suit_1:   0.0
             2th_board_card_suit_2:   0.0
             2th_board_card_suit_3:   0.0
             3th_board_card_rank_0:   0.0
             3th_board_card_rank_1:   0.0
             3th_board_card_rank_2:   0.0
             3th_board_card_rank_3:   0.0
             3th_board_card_rank_4:   0.0
             3th_board_card_rank_5:   0.0
             3th_board_card_rank_6:   0.0
             3th_board_card_rank_7:   0.0
             3th_board_card_rank_8:   0.0
             3th_board_card_rank_9:   0.0
            3th_board_card_rank_10:   0.0
            3th_board_card_rank_11:   0.0
            3th_board_card_rank_12:   0.0
             3th_board_card_suit_0:   0.0
             3th_board_card_suit_1:   0.0
             3th_board_card_suit_2:   0.0
             3th_board_card_suit_3:   0.0
             4th_board_card_rank_0:   0.0
             4th_board_card_rank_1:   0.0
             4th_board_card_rank_2:   0.0
             4th_board_card_rank_3:   0.0
             4th_board_card_rank_4:   0.0
             4th_board_card_rank_5:   0.0
             4th_board_card_rank_6:   0.0
             4th_board_card_rank_7:   0.0
             4th_board_card_rank_8:   0.0
             4th_board_card_rank_9:   0.0
            4th_board_card_rank_10:   0.0
            4th_board_card_rank_11:   0.0
            4th_board_card_rank_12:   0.0
             4th_board_card_suit_0:   0.0
             4th_board_card_suit_1:   0.0
             4th_board_card_suit_2:   0.0
             4th_board_card_suit_3:   0.0
          0th_player_card_0_rank_0:   0.0
          0th_player_card_0_rank_1:   0.0
          0th_player_card_0_rank_2:   1.0
          0th_player_card_0_rank_3:   0.0
          0th_player_card_0_rank_4:   0.0
          0th_player_card_0_rank_5:   0.0
          0th_player_card_0_rank_6:   0.0
          0th_player_card_0_rank_7:   0.0
          0th_player_card_0_rank_8:   0.0
          0th_player_card_0_rank_9:   0.0
         0th_player_card_0_rank_10:   0.0
         0th_player_card_0_rank_11:   0.0
         0th_player_card_0_rank_12:   0.0
          0th_player_card_0_suit_0:   0.0
          0th_player_card_0_suit_1:   0.0
          0th_player_card_0_suit_2:   1.0
          0th_player_card_0_suit_3:   0.0
          0th_player_card_1_rank_0:   0.0
          0th_player_card_1_rank_1:   0.0
          0th_player_card_1_rank_2:   0.0
          0th_player_card_1_rank_3:   0.0
          0th_player_card_1_rank_4:   0.0
          0th_player_card_1_rank_5:   0.0
          0th_player_card_1_rank_6:   0.0
          0th_player_card_1_rank_7:   0.0
          0th_player_card_1_rank_8:   1.0
          0th_player_card_1_rank_9:   0.0
         0th_player_card_1_rank_10:   0.0
         0th_player_card_1_rank_11:   0.0
         0th_player_card_1_rank_12:   0.0
          0th_player_card_1_suit_0:   0.0
          0th_player_card_1_suit_1:   1.0
          0th_player_card_1_suit_2:   0.0
          0th_player_card_1_suit_3:   0.0
          1th_player_card_0_rank_0:   0.0
          1th_player_card_0_rank_1:   0.0
          1th_player_card_0_rank_2:   0.0
          1th_player_card_0_rank_3:   0.0
          1th_player_card_0_rank_4:   0.0
          1th_player_card_0_rank_5:   0.0
          1th_player_card_0_rank_6:   0.0
          1th_player_card_0_rank_7:   0.0
          1th_player_card_0_rank_8:   0.0
          1th_player_card_0_rank_9:   0.0
         1th_player_card_0_rank_10:   0.0
         1th_player_card_0_rank_11:   0.0
         1th_player_card_0_rank_12:   0.0
          1th_player_card_0_suit_0:   0.0
          1th_player_card_0_suit_1:   0.0
          1th_player_card_0_suit_2:   0.0
          1th_player_card_0_suit_3:   0.0
          1th_player_card_1_rank_0:   0.0
          1th_player_card_1_rank_1:   0.0
          1th_player_card_1_rank_2:   0.0
          1th_player_card_1_rank_3:   0.0
          1th_player_card_1_rank_4:   0.0
          1th_player_card_1_rank_5:   0.0
          1th_player_card_1_rank_6:   0.0
          1th_player_card_1_rank_7:   0.0
          1th_player_card_1_rank_8:   0.0
          1th_player_card_1_rank_9:   0.0
         1th_player_card_1_rank_10:   0.0
         1th_player_card_1_rank_11:   0.0
         1th_player_card_1_rank_12:   0.0
          1th_player_card_1_suit_0:   0.0
          1th_player_card_1_suit_1:   0.0
          1th_player_card_1_suit_2:   0.0
          1th_player_card_1_suit_3:   0.0
          2th_player_card_0_rank_0:   0.0
          2th_player_card_0_rank_1:   0.0
          2th_player_card_0_rank_2:   0.0
          2th_player_card_0_rank_3:   0.0
          2th_player_card_0_rank_4:   0.0
          2th_player_card_0_rank_5:   0.0
          2th_player_card_0_rank_6:   0.0
          2th_player_card_0_rank_7:   0.0
          2th_player_card_0_rank_8:   0.0
          2th_player_card_0_rank_9:   0.0
         2th_player_card_0_rank_10:   0.0
         2th_player_card_0_rank_11:   0.0
         2th_player_card_0_rank_12:   0.0
          2th_player_card_0_suit_0:   0.0
          2th_player_card_0_suit_1:   0.0
          2th_player_card_0_suit_2:   0.0
          2th_player_card_0_suit_3:   0.0
          2th_player_card_1_rank_0:   0.0
          2th_player_card_1_rank_1:   0.0
          2th_player_card_1_rank_2:   0.0
          2th_player_card_1_rank_3:   0.0
          2th_player_card_1_rank_4:   0.0
          2th_player_card_1_rank_5:   0.0
          2th_player_card_1_rank_6:   0.0
          2th_player_card_1_rank_7:   0.0
          2th_player_card_1_rank_8:   0.0
          2th_player_card_1_rank_9:   0.0
         2th_player_card_1_rank_10:   0.0
         2th_player_card_1_rank_11:   0.0
         2th_player_card_1_rank_12:   0.0
          2th_player_card_1_suit_0:   0.0
          2th_player_card_1_suit_1:   0.0
          2th_player_card_1_suit_2:   0.0
          2th_player_card_1_suit_3:   0.0
          3th_player_card_0_rank_0:   0.0
          3th_player_card_0_rank_1:   0.0
          3th_player_card_0_rank_2:   0.0
          3th_player_card_0_rank_3:   0.0
          3th_player_card_0_rank_4:   0.0
          3th_player_card_0_rank_5:   0.0
          3th_player_card_0_rank_6:   0.0
          3th_player_card_0_rank_7:   0.0
          3th_player_card_0_rank_8:   0.0
          3th_player_card_0_rank_9:   0.0
         3th_player_card_0_rank_10:   0.0
         3th_player_card_0_rank_11:   0.0
         3th_player_card_0_rank_12:   0.0
          3th_player_card_0_suit_0:   0.0
          3th_player_card_0_suit_1:   0.0
          3th_player_card_0_suit_2:   0.0
          3th_player_card_0_suit_3:   0.0
          3th_player_card_1_rank_0:   0.0
          3th_player_card_1_rank_1:   0.0
          3th_player_card_1_rank_2:   0.0
          3th_player_card_1_rank_3:   0.0
          3th_player_card_1_rank_4:   0.0
          3th_player_card_1_rank_5:   0.0
          3th_player_card_1_rank_6:   0.0
          3th_player_card_1_rank_7:   0.0
          3th_player_card_1_rank_8:   0.0
          3th_player_card_1_rank_9:   0.0
         3th_player_card_1_rank_10:   0.0
         3th_player_card_1_rank_11:   0.0
         3th_player_card_1_rank_12:   0.0
          3th_player_card_1_suit_0:   0.0
          3th_player_card_1_suit_1:   0.0
          3th_player_card_1_suit_2:   0.0
          3th_player_card_1_suit_3:   0.0
          4th_player_card_0_rank_0:   0.0
          4th_player_card_0_rank_1:   0.0
          4th_player_card_0_rank_2:   0.0
          4th_player_card_0_rank_3:   0.0
          4th_player_card_0_rank_4:   0.0
          4th_player_card_0_rank_5:   0.0
          4th_player_card_0_rank_6:   0.0
          4th_player_card_0_rank_7:   0.0
          4th_player_card_0_rank_8:   0.0
          4th_player_card_0_rank_9:   0.0
         4th_player_card_0_rank_10:   0.0
         4th_player_card_0_rank_11:   0.0
         4th_player_card_0_rank_12:   0.0
          4th_player_card_0_suit_0:   0.0
          4th_player_card_0_suit_1:   0.0
          4th_player_card_0_suit_2:   0.0
          4th_player_card_0_suit_3:   0.0
          4th_player_card_1_rank_0:   0.0
          4th_player_card_1_rank_1:   0.0
          4th_player_card_1_rank_2:   0.0
          4th_player_card_1_rank_3:   0.0
          4th_player_card_1_rank_4:   0.0
          4th_player_card_1_rank_5:   0.0
          4th_player_card_1_rank_6:   0.0
          4th_player_card_1_rank_7:   0.0
          4th_player_card_1_rank_8:   0.0
          4th_player_card_1_rank_9:   0.0
         4th_player_card_1_rank_10:   0.0
         4th_player_card_1_rank_11:   0.0
         4th_player_card_1_rank_12:   0.0
          4th_player_card_1_suit_0:   0.0
          4th_player_card_1_suit_1:   0.0
          4th_player_card_1_suit_2:   0.0
          4th_player_card_1_suit_3:   0.0
          5th_player_card_0_rank_0:   0.0
          5th_player_card_0_rank_1:   0.0
          5th_player_card_0_rank_2:   0.0
          5th_player_card_0_rank_3:   0.0
          5th_player_card_0_rank_4:   0.0
          5th_player_card_0_rank_5:   0.0
          5th_player_card_0_rank_6:   0.0
          5th_player_card_0_rank_7:   0.0
          5th_player_card_0_rank_8:   0.0
          5th_player_card_0_rank_9:   0.0
         5th_player_card_0_rank_10:   0.0
         5th_player_card_0_rank_11:   0.0
         5th_player_card_0_rank_12:   0.0
          5th_player_card_0_suit_0:   0.0
          5th_player_card_0_suit_1:   0.0
          5th_player_card_0_suit_2:   0.0
          5th_player_card_0_suit_3:   0.0
          5th_player_card_1_rank_0:   0.0
          5th_player_card_1_rank_1:   0.0
          5th_player_card_1_rank_2:   0.0
          5th_player_card_1_rank_3:   0.0
          5th_player_card_1_rank_4:   0.0
          5th_player_card_1_rank_5:   0.0
          5th_player_card_1_rank_6:   0.0
          5th_player_card_1_rank_7:   0.0
          5th_player_card_1_rank_8:   0.0
          5th_player_card_1_rank_9:   0.0
         5th_player_card_1_rank_10:   0.0
         5th_player_card_1_rank_11:   0.0
         5th_player_card_1_rank_12:   0.0
          5th_player_card_1_suit_0:   0.0
          5th_player_card_1_suit_1:   0.0
          5th_player_card_1_suit_2:   0.0
          5th_player_card_1_suit_3:   0.0
    preflop_player_0_action_0_how_much:   0.0
      preflop_player_0_action_0_what_0:   0.0
      preflop_player_0_action_0_what_1:   0.0
      preflop_player_0_action_0_what_2:   0.0
    preflop_player_0_action_1_how_much:   0.0
      preflop_player_0_action_1_what_0:   0.0
      preflop_player_0_action_1_what_1:   0.0
      preflop_player_0_action_1_what_2:   0.0
    preflop_player_1_action_0_how_much:   0.0
      preflop_player_1_action_0_what_0:   0.0
      preflop_player_1_action_0_what_1:   0.0
      preflop_player_1_action_0_what_2:   0.0
    preflop_player_1_action_1_how_much:   0.0
      preflop_player_1_action_1_what_0:   0.0
      preflop_player_1_action_1_what_1:   0.0
      preflop_player_1_action_1_what_2:   0.0
    preflop_player_2_action_0_how_much:   0.0
      preflop_player_2_action_0_what_0:   0.0
      preflop_player_2_action_0_what_1:   0.0
      preflop_player_2_action_0_what_2:   0.0
    preflop_player_2_action_1_how_much:   0.0
      preflop_player_2_action_1_what_0:   0.0
      preflop_player_2_action_1_what_1:   0.0
      preflop_player_2_action_1_what_2:   0.0
    preflop_player_3_action_0_how_much:   0.0
      preflop_player_3_action_0_what_0:   0.0
      preflop_player_3_action_0_what_1:   0.0
      preflop_player_3_action_0_what_2:   0.0
    preflop_player_3_action_1_how_much:   0.0
      preflop_player_3_action_1_what_0:   0.0
      preflop_player_3_action_1_what_1:   0.0
      preflop_player_3_action_1_what_2:   0.0
    preflop_player_4_action_0_how_much:   0.0
      preflop_player_4_action_0_what_0:   0.0
      preflop_player_4_action_0_what_1:   0.0
      preflop_player_4_action_0_what_2:   0.0
    preflop_player_4_action_1_how_much:   0.0
      preflop_player_4_action_1_what_0:   0.0
      preflop_player_4_action_1_what_1:   0.0
      preflop_player_4_action_1_what_2:   0.0
    preflop_player_5_action_0_how_much:   0.0
      preflop_player_5_action_0_what_0:   0.0
      preflop_player_5_action_0_what_1:   1.0
      preflop_player_5_action_0_what_2:   200.0
    preflop_player_5_action_1_how_much:   0.0
      preflop_player_5_action_1_what_0:   0.0
      preflop_player_5_action_1_what_1:   0.0
      preflop_player_5_action_1_what_2:   0.0
       flop_player_0_action_0_how_much:   0.0
         flop_player_0_action_0_what_0:   0.0
         flop_player_0_action_0_what_1:   0.0
         flop_player_0_action_0_what_2:   0.0
       flop_player_0_action_1_how_much:   0.0
         flop_player_0_action_1_what_0:   0.0
         flop_player_0_action_1_what_1:   0.0
         flop_player_0_action_1_what_2:   0.0
       flop_player_1_action_0_how_much:   0.0
         flop_player_1_action_0_what_0:   0.0
         flop_player_1_action_0_what_1:   0.0
         flop_player_1_action_0_what_2:   0.0
       flop_player_1_action_1_how_much:   0.0
         flop_player_1_action_1_what_0:   0.0
         flop_player_1_action_1_what_1:   0.0
         flop_player_1_action_1_what_2:   0.0
       flop_player_2_action_0_how_much:   0.0
         flop_player_2_action_0_what_0:   0.0
         flop_player_2_action_0_what_1:   0.0
         flop_player_2_action_0_what_2:   0.0
       flop_player_2_action_1_how_much:   0.0
         flop_player_2_action_1_what_0:   0.0
         flop_player_2_action_1_what_1:   0.0
         flop_player_2_action_1_what_2:   0.0
       flop_player_3_action_0_how_much:   0.0
         flop_player_3_action_0_what_0:   0.0
         flop_player_3_action_0_what_1:   0.0
         flop_player_3_action_0_what_2:   0.0
       flop_player_3_action_1_how_much:   0.0
         flop_player_3_action_1_what_0:   0.0
         flop_player_3_action_1_what_1:   0.0
         flop_player_3_action_1_what_2:   0.0
       flop_player_4_action_0_how_much:   0.0
         flop_player_4_action_0_what_0:   0.0
         flop_player_4_action_0_what_1:   0.0
         flop_player_4_action_0_what_2:   0.0
       flop_player_4_action_1_how_much:   0.0
         flop_player_4_action_1_what_0:   0.0
         flop_player_4_action_1_what_1:   0.0
         flop_player_4_action_1_what_2:   0.0
       flop_player_5_action_0_how_much:   0.0
         flop_player_5_action_0_what_0:   0.0
         flop_player_5_action_0_what_1:   0.0
         flop_player_5_action_0_what_2:   0.0
       flop_player_5_action_1_how_much:   0.0
         flop_player_5_action_1_what_0:   0.0
         flop_player_5_action_1_what_1:   0.0
         flop_player_5_action_1_what_2:   0.0
       turn_player_0_action_0_how_much:   0.0
         turn_player_0_action_0_what_0:   0.0
         turn_player_0_action_0_what_1:   0.0
         turn_player_0_action_0_what_2:   0.0
       turn_player_0_action_1_how_much:   0.0
         turn_player_0_action_1_what_0:   0.0
         turn_player_0_action_1_what_1:   0.0
         turn_player_0_action_1_what_2:   0.0
       turn_player_1_action_0_how_much:   0.0
         turn_player_1_action_0_what_0:   0.0
         turn_player_1_action_0_what_1:   0.0
         turn_player_1_action_0_what_2:   0.0
       turn_player_1_action_1_how_much:   0.0
         turn_player_1_action_1_what_0:   0.0
         turn_player_1_action_1_what_1:   0.0
         turn_player_1_action_1_what_2:   0.0
       turn_player_2_action_0_how_much:   0.0
         turn_player_2_action_0_what_0:   0.0
         turn_player_2_action_0_what_1:   0.0
         turn_player_2_action_0_what_2:   0.0
       turn_player_2_action_1_how_much:   0.0
         turn_player_2_action_1_what_0:   0.0
         turn_player_2_action_1_what_1:   0.0
         turn_player_2_action_1_what_2:   0.0
       turn_player_3_action_0_how_much:   0.0
         turn_player_3_action_0_what_0:   0.0
         turn_player_3_action_0_what_1:   0.0
         turn_player_3_action_0_what_2:   0.0
       turn_player_3_action_1_how_much:   0.0
         turn_player_3_action_1_what_0:   0.0
         turn_player_3_action_1_what_1:   0.0
         turn_player_3_action_1_what_2:   0.0
       turn_player_4_action_0_how_much:   0.0
         turn_player_4_action_0_what_0:   0.0
         turn_player_4_action_0_what_1:   0.0
         turn_player_4_action_0_what_2:   0.0
       turn_player_4_action_1_how_much:   0.0
         turn_player_4_action_1_what_0:   0.0
         turn_player_4_action_1_what_1:   0.0
         turn_player_4_action_1_what_2:   0.0
       turn_player_5_action_0_how_much:   0.0
         turn_player_5_action_0_what_0:   0.0
         turn_player_5_action_0_what_1:   0.0
         turn_player_5_action_0_what_2:   0.0
       turn_player_5_action_1_how_much:   0.0
         turn_player_5_action_1_what_0:   0.0
         turn_player_5_action_1_what_1:   0.0
         turn_player_5_action_1_what_2:   0.0
      river_player_0_action_0_how_much:   0.0
        river_player_0_action_0_what_0:   0.0
        river_player_0_action_0_what_1:   0.0
        river_player_0_action_0_what_2:   0.0
      river_player_0_action_1_how_much:   0.0
        river_player_0_action_1_what_0:   0.0
        river_player_0_action_1_what_1:   0.0
        river_player_0_action_1_what_2:   0.0
      river_player_1_action_0_how_much:   0.0
        river_player_1_action_0_what_0:   0.0
        river_player_1_action_0_what_1:   0.0
        river_player_1_action_0_what_2:   0.0
      river_player_1_action_1_how_much:   0.0
        river_player_1_action_1_what_0:   0.0
        river_player_1_action_1_what_1:   0.0
        river_player_1_action_1_what_2:   0.0
      river_player_2_action_0_how_much:   0.0
        river_player_2_action_0_what_0:   0.0
        river_player_2_action_0_what_1:   0.0
        river_player_2_action_0_what_2:   0.0
      river_player_2_action_1_how_much:   0.0
        river_player_2_action_1_what_0:   0.0
        river_player_2_action_1_what_1:   0.0
        river_player_2_action_1_what_2:   0.0
      river_player_3_action_0_how_much:   0.0
        river_player_3_action_0_what_0:   0.0
        river_player_3_action_0_what_1:   0.0
        river_player_3_action_0_what_2:   0.0
      river_player_3_action_1_how_much:   0.0
        river_player_3_action_1_what_0:   0.0
        river_player_3_action_1_what_1:   0.0
        river_player_3_action_1_what_2:   0.0
      river_player_4_action_0_how_much:   0.0
        river_player_4_action_0_what_0:   0.0
        river_player_4_action_0_what_1:   0.0
        river_player_4_action_0_what_2:   0.0
      river_player_4_action_1_how_much:   0.0
        river_player_4_action_1_what_0:   0.0
        river_player_4_action_1_what_1:   0.0
        river_player_4_action_1_what_2:   0.0
      river_player_5_action_0_how_much:   0.0
        river_player_5_action_0_what_0:   0.0
        river_player_5_action_0_what_1:   0.0
        river_player_5_action_0_what_2:   0.0
      river_player_5_action_1_how_much:   0.0
        river_player_5_action_1_what_0:   0.0
        river_player_5_action_1_what_1:   0.0
        river_player_5_action_1_what_2:   0.0
                          button_index:   1.0
