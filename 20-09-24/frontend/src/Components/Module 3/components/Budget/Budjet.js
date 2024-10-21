import React, { useState, useEffect } from 'react';
import { Card, Button, Modal, Form, ProgressBar } from 'react-bootstrap';
import axios from 'axios';
import './budjet.css';
import { FaPlus } from "react-icons/fa";

function Budjet() {
    const [budgets, setBudgets] = useState([]); 
    const [showModal, setShowModal] = useState(false);
    const [newBudgetName, setNewBudgetName] = useState('');
    const [newBudgetAmount, setNewBudgetAmount] = useState('');

    const URL = "http://localhost:3001/api/v1/";

    // Fetch budgets on component mount
    useEffect(() => {
        fetchBudgets();
    }, []);

    const fetchBudgets = async () => {
        const userId = localStorage.getItem("userId");
        try {
            const response = await axios.get(`${URL}get-budgets/${userId}`);
            console.log(response.data);
            if (response.data && Array.isArray(response.data)) {
                setBudgets(response.data);
            } else {
                setBudgets([]);
            }
        } catch (error) {
            console.error('Error fetching budgets:', error);
            setBudgets([]);
        }
    };

    // Function to handle budget creation
    const handleCreateBudget = async () => {
        if (!newBudgetName || !newBudgetAmount) return;

        try {
            const userId = localStorage.getItem("userId");
            const response = await axios.post(`${URL}create-budget`, {
                userId: userId,
                budget_name: newBudgetName,
                amount: newBudgetAmount
            });
            setBudgets([...budgets, response.data.budget]);
            setShowModal(false);
            fetchBudgets();
            setNewBudgetName('');
            setNewBudgetAmount('');
        } catch (error) {
            alert(error.response.data.message)
            console.error('Error creating budget:', error);
        }
    };

    return (
        <div className="InnerLayout">
            <h2 className="budget-header my-4">Budgets</h2>
            <div className="row">
                {/* Render each budget card */}
                {budgets.map(budget => {
                    // Calculate total expenses and count
                    const totalSpent = budget.expenses ? budget.expenses.reduce((acc, expense) => acc + expense.amount, 0) : 0;
                    const expenseCount = budget.expenses ? budget.expenses.length : 0;
                    const remainingAmount = budget.amount - totalSpent;

                    // Calculate percentage spent
                    const percentageSpent = budget.amount ? (totalSpent / budget.amount) * 100 : 0;

                    return (
                        <div className="col-md-4 mb-3 " key={budget._id}>
                            <Card className="budget-card">
                                <Card.Body>
                                    <div className="budget-amount">
                                        <span className="budget-amount-symbol">₹</span>
                                        {budget.amount}
                                    </div>
                                    <div className="budget-header">
                                    <Card.Title className="budget-card-title">{budget.budget_name.toUpperCase()}</Card.Title>
                                    <div className="budget-details">
                                        <p>{expenseCount} item</p>
                                    </div>
                                    </div>
                                    {/* Progress Bar Wrapper */}
                                    <div className="progress-bar-wrapper">
                                        <div className="progress-bar-labels">
                                            <span>₹{totalSpent} Spend</span>
                                            <span>₹{remainingAmount} Remaining</span>
                                        </div>
                                        <ProgressBar now={percentageSpent} />
                                    </div>
                                </Card.Body>
                            </Card>
                        </div>
                    );
                })}

                {/* Card for adding a new budget */}
                <div className="col-md-4 mb-3">
                    <Card className="budget-card" onClick={(e) => { e.stopPropagation(); setShowModal(true); }} style={{ cursor: 'pointer' }}>
                        <Card.Body>
                            <Card.Title className="budget-card-add-title"><FaPlus className="budget-card-add-icon"/><br/>Create New Budget</Card.Title>
                        </Card.Body>
                    </Card>
                </div>
            </div>

            {/* Modal for creating a new budget */}
            <Modal show={showModal} onHide={() => setShowModal(false)}>
                <Modal.Header  className='budget-model-headder'closeButton>
                    <Modal.Title>Create Budget</Modal.Title>
                </Modal.Header>
                <Modal.Body>
                    <Form>
                        <Form.Group className="mb-3" controlId="budgetName">
                            <Form.Label>Budget Name</Form.Label>
                            <Form.Control 
                                type="text" 
                                placeholder="Enter budget name" 
                                required={true}
                                value={newBudgetName}
                                onChange={(e) => setNewBudgetName(e.target.value)}
                            />
                        </Form.Group>
                        <Form.Group className="mb-3" controlId="budgetAmount">
                            <Form.Label>Budget Amount</Form.Label>
                            <Form.Control 
                                type="number" 
                                placeholder="Enter budget amount" 
                                required={true}
                                value={newBudgetAmount}
                                onChange={(e) => setNewBudgetAmount(e.target.value)}
                            />
                        </Form.Group>
                    </Form>
                </Modal.Body>
                <Modal.Footer>
                    <Button variant="secondary" onClick={() => {
                       setShowModal(false)
                       setNewBudgetName('')
                       setNewBudgetAmount('')
                    }
                     }>
                        Close
                    </Button>
                    <Button variant="primary" onClick={handleCreateBudget}>
                        Save Budget
                    </Button>
                </Modal.Footer>
            </Modal>
        </div>
    );
}

export default Budjet;
