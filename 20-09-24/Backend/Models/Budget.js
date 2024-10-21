const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const BudgetExpenseSchema = new Schema({
    name: {
        type: String,
        required: true,
        trim: true,
        maxLength: 50
    },
    date: {
        type: Date,
        required: true,
        trim: true
    },
    amount: {
        type: Number,
        required: true,
        maxLength: 20,
        trim: true
    }
}, { timestamps: true });

const BudgetEntrySchema = new Schema({
    id: {
        type: String, // or ObjectId if you want to use Mongoose's ObjectId
        required: true,
        // unique: true, // Ensure each budget entry has a unique identifier
    },
    budget_name: {
        type: String,
        required: true,
        trim: true,
        maxLength: 100
    },
    amount: {
        type: Number,
        required: true,
        maxLength: 20,
        trim: true
    },
    expenses: [BudgetExpenseSchema]  // Array of expenses within this budget
}, { _id: true, timestamps: true });

const UserBudgetSchema = new Schema({
    userId: {
        type: Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    budget_list: [BudgetEntrySchema]  // Array of budget entries
}, { timestamps: true });

module.exports = mongoose.model('Budget', UserBudgetSchema);
