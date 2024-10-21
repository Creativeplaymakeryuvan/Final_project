const ExpenseSchema = require('../Models/expense')
const mindee = require('mindee');

// exports.addExpense = async (req, res) => {
//     const {title, amount, category, description, date} = req.body

//     const expense = ExpenseScema({
//         title,
//         amount,
//         category,
//         description,
//         date
//     })

//     try{
//         if(!title || !category || !description || !date){
//             return res.status(400).json({ messaage: 'All fields are require!'})
//         }
//         if(amount <= 0 || !amount === 'number'){
//             return res.status(400).json({ messaage: 'Amount must be a positive'})
//         }
//         await expense.save()
//         res.status(200).json({ message: 'Expense Added' })
//     }catch(error){
//         res.status(500).json({ message: 'Server Error' })
//     }
//     console.log(expense)
// }

exports.addExpense = async (req, res) => {
    const { title, amount, category, description, date, userId } = req.body;

    try {
        // Validate input fields
        if (!title || !category || !description || !date) {
            return res.status(400).json({ message: 'All fields are required!' });
        }
        if (amount <= 0 || !amount == 'number') {
            return res.status(400).json({ message: 'Amount must be a positive number' });
        }

        let expenseDoc = await ExpenseSchema.findOne({ userId });

        if (!expenseDoc) {
            expenseDoc = new ExpenseSchema({
                userId,
                expenses: []
            });
        }

        expenseDoc.expenses.push({ title, amount, category, description, date });
        await expenseDoc.save();
        res.status(201).json({ message: 'Expense Added', expense: expenseDoc });
    } catch (error) {
        res.status(500).json({ message: 'Server Error', error: error.message });
    }
};

exports.getExpenses = async (req, res) => {
    try {
        const { userId } = req.params
        const expenseDoc = await ExpenseSchema.findOne({ userId });

        if (!expenseDoc) {
            return res.status(404).json({ message: 'No expense records found for this user' });
        }

        res.status(200).json(expenseDoc.expenses)
    }
    catch (error) {
        res.status(500).json({ message: 'Server Error' })
    }
}


exports.deleteExpense = async (req, res) => {
    const { userId, expenseId } = req.params;

    try {
        const expenseDoc = await ExpenseSchema.findOne({ userId });

        if (!expenseDoc) {
            return res.status(404).json({ message: 'Expense document not found for this user' });
        }

        // Filter out the income entry to be deleted
        const initialLength = expenseDoc.expenses.length;
        expenseDoc.expenses = expenseDoc.expenses.filter(expense => expense._id.toString() !== expenseId);

        if (expenseDoc.expenses.length === initialLength) {
            return res.status(404).json({ message: 'Expense entry not found' });
        }

        await expenseDoc.save();

        res.status(200).json({ message: 'Expense entry deleted successfully' });
    } catch (error) {
        res.status(500).json({ message: 'Server Error', error: error.message });
    }
}

exports.updateExpense = async (req, res) => {
    try {
        const { userId, expenseId } = req.params;
        const updateFields = req.body;

        const expenseDoc = await ExpenseSchema.findOne({ userId });

        if (!expenseDoc) {
            return res.status(404).json({ message: 'Expense document not found for this user' });
        }

        const expenseEntry = expenseDoc.expenses.id(expenseId);
        if (!expenseEntry) {
            return res.status(404).json({ message: 'Expense not found' });
        }

        Object.assign(expenseEntry, updateFields);

        await expenseDoc.save();

        return res.status(200).json({ message: 'Expense updated successfully', data: expenseEntry });
    } catch (error) {
        return res.status(500).json({ message: 'Error updating expense', error: error.message });
    }
};



exports.processBillImage = async (req, res) => {
    const mindeeClient = new mindee.Client({ apiKey: '1af6c810e08b28f6083bcfa49e0e6091' });
    try {
        const fileBuffer = req.file.buffer;

        const inputSource = mindeeClient.docFromBuffer(fileBuffer, req.file.originalname);

        const apiResponse = await mindeeClient.parse(
            mindee.product.ReceiptV5,
            inputSource
        );

        res.status(200).json(apiResponse.document);
    } catch (error) {
        console.error('Error processing bill image:', error);
        res.status(500).json({ error: 'Failed to process the bill image' });
    }
};

exports.processInvoice = async (req, res) => {
    const mindeeClient = new mindee.Client({ apiKey: "b4b3db41b2557709c3922b111cf809e1" });

    try {
        const inputSource = mindeeClient.docFromPath("/path/to/the/file.ext");

        const apiResponse = mindeeClient.parse(
            mindee.product.InvoiceV4,
            inputSource
        );
        res.status(200).json(apiResponse.document);
    }
    catch (e) {
        console.error('Error processing bill image:', error);
        res.status(500).json({ error: 'Failed to process the Invoice' });
    }

}