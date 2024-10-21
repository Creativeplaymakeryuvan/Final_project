// const mongoose = require('mongoose')

// const IncomeEntrySchema = new mongoose.Schema({
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
//     // type: {
//     //     type: String,
//     //     default: "income"
//     // },
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
//     }
// }, {_id: false})

// const IncomeSchema = new mongoose.Schema({
//     userId: {
//         type: mongoose.Schema.Types.ObjectId,
//         ref: 'User',
//         required: true,
//         unique: true
//     },
//     Incomes: [IncomeEntrySchema]
// }, { timestamps: true });
// module.exports = mongoose.model('In', IncomeSchema)

const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const IncomeEntrySchema = new Schema({
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
        default: "income"
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

const IncomeSchema = new Schema({
    userId: {
        type: Schema.Types.ObjectId,
        ref: 'User',
        required: true,
        unique: true
    },
    incomes: [IncomeEntrySchema]
}, { timestamps: true });

module.exports = mongoose.model('In', IncomeSchema);
