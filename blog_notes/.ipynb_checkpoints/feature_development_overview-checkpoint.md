# Building a Feature Development Engine for Player Profiling

## Core Idea
Why did I chose to build this, and what did I hope to accomplish?

## Sections

### Why did I want to do this

- The amount of money getting spent in football nowadays is ridiculous
- From Trevor Francis being the first 'million pound man'in football with his 1979 transfer from Birmingham to Forest in 79 (though only $999,999 was paid by Brian Clough to avoid the label) prices on players have been increasing at an exponential rate
- ![British Record Transfers over time](../notebooks/research/introduction/record_transfer_history.png)
- Whilst it looks like exponential growth, the truth is actually more itneresting.  You can track the history of football through the record transfer fees
- 60s -> gate entry, 79 -> tv money !(need to check this), 96 -> Prem money, 01 -> champions league globalisation, 10's -> oil state money, 2018 + sovereign wealth and financialization
- Graph is a stair case, showing a systematic reset of the financial ceiling of football
- Realistically there is a finite limit on soverign states that will be able to invest in footabll clubs
- Very few clubs will be able to compete for the top value players
- less financially abundant clubs need to find ways to compete
- Finding undervalued/overlooked talent
- To do this we need to be able to drill down into what defines a player, both in your existing squads and what you looking to recruit
- Needs to be easy to extend and develop against
- Needs to output results in an understandable way - come back to core principle that if no one can understand you any analysis is waster
- Using statsbomb data

### Setting up the parameters
#### 