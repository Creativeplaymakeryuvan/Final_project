const UserBudget = require('../Models/Budget');
const mongoose = require('mongoose');

// Create a new budget
exports.addBudget = async (req, res) => {
    try {
        const { userId, budget_name, amount } = req.body;

        // Validate the amount
        if (amount <= 0) {
            return res.status(400).json({ message: "Budget amount must be greater than 0." });
        }

        // Find the user budget document
        let userBudget = await UserBudget.findOne({ userId });

        // If no document exists, create one
        if (!userBudget) {
            userBudget = new UserBudget({ userId, budget_list: [] });
        }

        // Create a unique id for the new budget entry
        const budgetId = new mongoose.Types.ObjectId();

        // Add the new budget to the budget_list array
        userBudget.budget_list.push({
            id: budgetId.toString(),  // Ensure the id is a string
            budget_name,
            amount,
            expenses: []
        });

        await userBudget.save();
        res.status(201).json({ message: "Budget added successfully", budget: userBudget });
    } catch (error) {
        res.status(500).json({ message: "Server Error", error: error.message });
    }
};


exports.getBudgets = async (req, res) => {
    try {
        const { userId } = req.params;

        // Find the user's budget list
        const userBudget = await UserBudget.findOne({ userId });

        if (!userBudget) {
            return res.status(404).json({ message: "No budgets found for this user" });
        }

        res.status(200).json(userBudget.budget_list);
    } catch (error) {
        res.status(500).json({ message: "Server Error", error: error.message });
    }
};


exports.deleteBudget = async (req, res) => {
    try {
        const { userId, budgetId } = req.params;

        // Find the user's budget document
        const userBudget = await UserBudget.findOne({ userId });

        if (!userBudget) {
            return res.status(404).json({ message: "User budget not found" });
        }

        // Check if the budgetId exists in the budget list
        const budgetExists = userBudget.budget_list.some(budget => budget._id.toString() === budgetId); // Use `id` field for comparison
        
        if (!budgetExists) {
            return res.status(404).json({ message: "Budget not found" });
        }

        // Remove the specific budget by filtering the budget list
        userBudget.budget_list = userBudget.budget_list.filter(budget => budget._id.toString() !== budgetId); // Use `id` field for filtering

        // Save the updated budget list to the database
        await userBudget.save();

        // Return success response
        res.status(200).json({ message: "Budget deleted successfully" });
    } catch (error) {
        // Handle server errors
        res.status(500).json({ message: "Server Error", error: error.message });
    }
};



exports.addExpenseToBudget = async (req, res) => {
    try {
        const { userId, budgetId } = req.params;
        const { name, date, amount } = req.body;

        // Find the user's budget list
        const userBudget = await UserBudget.findOne({ userId });

        if (!userBudget) {
            return res.status(404).json({ message: "User not found" });
        }

        // Find the specific budget in the budget list
        const budget = userBudget.budget_list.id(budgetId);
        if (!budget) {
            return res.status(404).json({ message: "Budget not found" });
        }

        // Add the new expense to the budget
        budget.expenses.push({ name, date, amount });

        await userBudget.save();
        res.status(201).json({ message: "Expense added successfully", budget });
    } catch (error) {
        res.status(500).json({ message: "Server Error", error: error.message });
    }
};

exports.getExpensesForBudget = async (req, res) => {
    try {
        const { userId, budgetId } = req.params;

        // Find the user's budget list
        const userBudget = await UserBudget.findOne({ userId });

        if (!userBudget) {
            return res.status(404).json({ message: "User not found" });
        }

        // Find the specific budget
        const budget = userBudget.budget_list.id(budgetId);
        if (!budget) {
            return res.status(404).json({ message: "Budget not found" });
        }

        res.status(200).json(budget.expenses);
    } catch (error) {
        res.status(500).json({ message: "Server Error", error: error.message });
    }
};

exports.deleteBudgetExpense = async (req, res) => {
    try {
        const { userId, budgetId, expenseId } = req.params;

        // Find the user's budget list
        const userBudget = await UserBudget.findOne({ userId });

        if (!userBudget) {
            return res.status(404).json({ message: "User not found" });
        }

        // Find the specific budget
        const budget = userBudget.budget_list.id(budgetId);
        if (!budget) {
            return res.status(404).json({ message: "Budget not found" });
        }

        // Remove the specific expense
        const updatedExpenses = budget.expenses.filter(expense => expense._id.toString() !== expenseId);
        budget.expenses = updatedExpenses;

        await userBudget.save();
        res.status(200).json({ message: "Expense deleted successfully" });
    } catch (error) {
        res.status(500).json({ message: "Server Error", error: error.message });
    }
};



