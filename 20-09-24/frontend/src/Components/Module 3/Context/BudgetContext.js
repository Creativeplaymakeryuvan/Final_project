import React, { createContext, useContext, useState } from "react";
import axios from "axios";

// Base URL for API calls
const URL = "http://localhost:3001/api/v1/";
const BudgetContext = createContext();

// BudgetProvider component to encapsulate the budgeting logic and state management
export const BudgetProvider = ({ children }) => {
    const [budgets, setBudgets] = useState([]); // State for storing all budgets
    const [budgetExpenses, setBudgetExpenses] = useState([]); // State for storing expenses of a specific budget
    const [error, setError] = useState(null); // State for managing errors

    const createBudget = async (budget) => {
        const userId = localStorage.getItem("userId"); // Retrieve user ID from localStorage
        try {
            await axios.post(`${URL}create-budget`, { userId, ...budget }); // Send request to create budget
            alert('Budget created');
            getBudgets(); // Refresh budgets after creation
        } catch (err) {
            alert('Budget not created');
            setError(err.response?.data?.message || "An error occurred"); // Set error message
        }
    };

    // Get all budgets for a user
    const getBudgets = async () => {
        const userId = localStorage.getItem("userId"); // Retrieve user ID from localStorage
        try {
            const response = await axios.get(`${URL}get-budgets/${userId}`); // Fetch user's budgets
            setBudgets(response.data); // Update budgets state
        } catch (err) {
            setError(err.response?.data?.message || "An error occurred"); // Set error message
        }
    };

    // Add an expense to a specific budget
    const addBudgetExpense = async (budgetId, expense) => {
        try {
            await axios.post(`${URL}add-budget-expense/${budgetId}`, expense); // Send request to add expense
            alert('Budget expense added');
            getBudgetExpenses(budgetId); // Refresh budget expenses after adding
        } catch (err) {
            alert('Budget expense not added');
            setError(err.response?.data?.message || "An error occurred"); // Set error message
        }
    };

    // Get all expenses for a specific budget
    const getBudgetExpenses = async (budgetId) => {
        try {
            const response = await axios.get(`${URL}get-budget-expenses/${budgetId}`); // Fetch budget expenses
            setBudgetExpenses(response.data); // Update budget expenses state
        } catch (err) {
            setError(err.response?.data?.message || "An error occurred"); // Set error message
        }
    };

    // Delete an expense from a specific budget
    const deleteBudgetExpense = async (userId, budgetId, expenseId) => {
        try {
            await axios.delete(`${URL}delete-budget-expense/${userId}/${budgetId}/${expenseId}`); // Send request to delete expense
            alert('Budget expense deleted');
            getBudgets(); // Refresh budgets after deletion
            getBudgetExpenses(budgetId); // Refresh budget expenses after deletion
        } catch (err) {
            alert('Budget expense not deleted');
            setError(err.response?.data?.message || "An error occurred"); // Set error message
        }
    };

    // Calculate total expenses for all budgets
    const totalBudgets = () => {
        return budgets.reduce((total, budget) => {
            const budgetTotal = budget.expenses.reduce((sum, expense) => sum + expense.amount, 0);
            return total + budgetTotal; // Sum up total expenses for each budget
        }, 0);
    };

    // Calculate total amount of expenses in a budget
    const totalBudgetExpenses = () => {
        return budgetExpenses.reduce((total, expense) => total + expense.amount, 0); // Sum up total expenses in the selected budget
    };

    return (
        <BudgetContext.Provider value={{
            createBudget,
            getBudgets,
            budgets,
            addBudgetExpense,
            deleteBudgetExpense,
            totalBudgets,
            budgetExpenses,
            getBudgetExpenses,
            totalBudgetExpenses,
            error,
        }}>
            {children} {/* Render children components that can access this context */}
        </BudgetContext.Provider>
    );
};

// Custom hook to use the BudgetContext
export const useBudgetContext = () => {
    return useContext(BudgetContext);
};
