// const mongoose = require('mongoose')

// const ExpenseSchema = new mongoose.Schema({
//     title: {
//         type: String,
//         required: true,
//         trim: true,
//         maxLength: 50
//     },
//     amount: {
//         type: Number,
//         required: true,
//         maxLength: 20,
//         trim: true
//     },
//     type: {
//         type: String,
//         default: "income"
//     },
//     date: {
//         type: Date,
//         required: true,
//         trim: true
//     },
//     category: {
//         type: String,
//         required: true,
//         trim: true
//     },
//     description: {
//         type: String,
//         required: true,
//         maxLength: 20,
//         trim: true
//     },
// }, {timestamps: true})

// module.exports = mongoose.model('Expense', ExpenseSchema)

const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const ExpenseEntrySchema = new Schema({
    amount: {
        type: Number,
        required: true,
        maxLength: 20,
        trim: true
    },
    title: {
        type: String,
        required: true,
        trim: true,
        maxLength: 50
    },
    type: {
        type: String,
        default: "expense"
    },
    date: {
        type: Date,
        required: true,
        trim: true
    },
    description: {
        type: String,
        required: true,
        maxLength: 200,
        trim: true
    },
    category: {
        type: String,
        required: true,
        trim: true
    }
}, { timestamps: true });

const ExpenseSchema = new Schema({
    userId: {
        type: Schema.Types.ObjectId,
        ref: 'User',
        required: true,
        unique: true
    },
    expenses: [ExpenseEntrySchema]
}, { timestamps: true });

module.exports = mongoose.model('Expense', ExpenseSchema);