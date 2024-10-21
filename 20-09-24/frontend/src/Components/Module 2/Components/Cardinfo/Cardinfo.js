import React, { useEffect, useState } from 'react';
import './cardinfo.css';
import { useGlobalContext } from '../../Context/globalContext';

function Cardinfo({ totalMonthlyIncome, totalMonthlyExpenses, balance, totalSavings, totalSavingsbefore, flag }) {
    const { totalExpenses, totalIncomes, totalBalance, getIncomes, getExpenses } = useGlobalContext();
    
    const [currentCard, setCurrentCard] = useState('activeBalance'); 
    const [transitioning, setTransitioning] = useState(false);
    const [showBorrowedCard, setShowBorrowedCard] = useState(false);
    const [adjustedTotalSavings, setAdjustedTotalSavings] = useState(totalSavings);

    useEffect(() => {
        if (totalSavings < 0) {
            setShowBorrowedCard(true);
            setAdjustedTotalSavings(0);
        } else {
            setShowBorrowedCard(false);
            setAdjustedTotalSavings(totalSavings);
        }
    }, [totalSavings]);

    useEffect(() => {
        const interval = setInterval(() => {
            if (!transitioning) {
                setTransitioning(true); 
                setTimeout(() => {
                    if (showBorrowedCard) {
                        switch (currentCard) {
                            case 'activeBalance':
                                setCurrentCard('previousBalance');
                                break;
                            case 'previousBalance':
                                setCurrentCard('totalSavings');
                                break;
                            case 'totalSavings':
                                setCurrentCard('borrowed');
                                break;
                            case 'borrowed':
                                setCurrentCard('activeBalance');
                                break;
                            default:
                                setCurrentCard('activeBalance');
                        }
                    } else {
                        switch (currentCard) {
                            case 'activeBalance':
                                setCurrentCard('previousBalance');
                                break;
                            case 'previousBalance':
                                setCurrentCard('totalSavings');
                                break;
                            default:
                                setCurrentCard('activeBalance');
                        }
                    }
                    setTransitioning(false);
                }, 1000);
            }
        }, 4000);

        return () => clearInterval(interval);
    }, [currentCard, transitioning, showBorrowedCard]);

    useEffect(() => {
        getIncomes();
        getExpenses();
    }, []);

    return (
        <div className="cardinfo-container">
            <div className="card">
                <div>
                    <h2 className="title title-in">Total Income</h2>
                    <h2 className="amount amount-in">₹{flag ? totalMonthlyIncome : totalIncomes()}</h2>
                </div>
            </div>
            <div className="card">
                <div>
                    <h2 className="title title-ex">Total Expense</h2>
                    <h2 className="amount amount-ex">₹{flag ? totalMonthlyExpenses : totalExpenses()}</h2>
                </div>
            </div>
            <div className="card combined-card">
                <div className={`card-content ${currentCard === 'activeBalance' ? 'fade-in' : ''}`}>
                    <h2 className="title">Active Balance</h2>
                    <h2 className="amount">₹{flag ? balance : totalBalance()}</h2>
                </div>

                <div className={`card-content ${currentCard === 'previousBalance' ? 'fade-in' : ''}`}>
                    <h2 className="title">Previous Savings</h2>
                    <h2 className="amount">₹{totalSavingsbefore}</h2>
                </div>

                <div className={`card-content ${currentCard === 'totalSavings' ? 'fade-in' : ''}`}>
                    <h2 className="title">Total Savings</h2>
                    <h2 className="amount">₹{adjustedTotalSavings}</h2>
                </div>

                {showBorrowedCard && (
                    <div className={`card-content ${currentCard === 'borrowed' ? 'fade-in' : ''}`}>
                        <h2 className="title">Borrowed</h2>
                        <h2 className="amount">₹{Math.abs(totalSavings)}</h2> 
                    </div>
                )}
            </div>
        </div>
    );
}

export default Cardinfo;
