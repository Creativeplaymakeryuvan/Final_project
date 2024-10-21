const IncomeScema = require('../Models/income')

// exports.addIncome = async (req, res) => {
//     const { userId } = req.params;
//     const { title, amount, category, description, date  } = req.body

//     // const income = IncomeScema({
//     //     title,
//     //     amount,
//     //     category,
//     //     description,
//     //     date,
//     //     userId
//     // })

//     try {
//         if (!title || !category || !description || !date) {
//             return res.status(400).json({ messaage: 'All fields are require!' })
//         }
//         if (amount <= 0 || !amount === 'number') {
//             return res.status(400).json({ messaage: 'Amount must be a positive' })
//         }

//         // Finding the user
//         let incomeDoc = await IncomeScema.findOne({ userId });
//         if(!incomeDoc){
//             incomeDoc = new IncomeScema({
//                 userId,
//                 incomes: []
//             });
//         }
//         incomeDoc.Incomes.push({  title,amount,category,description,date })
//         await incomeDoc.save();
//         res.status(200).json({ message: 'Income Added' })
//     } catch (error) {
//         res.status(500).json({ message: 'Server Error' })
//     }
//     console.log(Income)
// }

exports.addIncome = async (req, res) => {
    const { title, amount, category, description, date, userId } = req.body;

    try {
        // Validate input fields
        if (!title || !category || !description || !date) {
            return res.status(400).json({ message: 'All fields are required!' });
        }
        if (amount <= 0 || !amount == 'number') {
            return res.status(400).json({ message: 'Amount must be a positive number' });
        }

        let incomeDoc = await IncomeScema.findOne({ userId });

        if (!incomeDoc) {
            incomeDoc = new IncomeScema({
                userId,
                incomes: []
            });
        }

        incomeDoc.incomes.push({ title, amount, category, description, date});

        await incomeDoc.save();

        res.status(201).json({ message: 'Income Added', income: incomeDoc });
    } catch (error) {
        res.status(500).json({ message: 'Server Error', error: error.message });
    }
};

exports.getIncomes = async (req, res) => {
    try {
        const { userId } = req.params
        const incomeDoc = await IncomeScema.findOne({ userId });

        if(!incomeDoc){
            return res.status(404).json({ message: 'No income records found for this user' });
        }

        res.status(200).json(incomeDoc.incomes)
    }
    catch (error) {
        res.status(500).json({ message: 'Server Error' })
    }
}

// exports.deleteIncomes = async (req, res) => {
//     const { id } = req.params;
//     res.status(200).json({ id })
//     IncomeScema.findByIdAndDelete(id)
//         .then((income) => {
//             res.status(200).json({ message: 'Income Deleted' })
//         })
//         .catch((err) => {
//             res.status(500).json({ message: 'Server Error' })
//         })
// }

exports.deleteIncomes = async (req, res) => {
    const { userId, incomeId } = req.params;
    
    try {
        const incomeDoc = await IncomeScema.findOne({ userId });

        if (!incomeDoc) {
            return res.status(404).json({ message: 'Income document not found for this user' });
        }

        // Filter out the income entry to be deleted
        const initialLength = incomeDoc.incomes.length;
        incomeDoc.incomes = incomeDoc.incomes.filter(income => income._id.toString() !== incomeId);

        if (incomeDoc.incomes.length === initialLength) {
            return res.status(404).json({ message: 'Income entry not found' });
        }

        await incomeDoc.save();

        res.status(200).json({ message: 'Income entry deleted successfully' });
    } catch (error) {
        res.status(500).json({ message: 'Server Error', error: error.message });
    }
};

exports.updateIncome = async (req, res) => {
    try {
      const { userId, incomeId } = req.params;     
      const updateFields = req.body;
  
      const incomeDoc = await IncomeScema.findOne({ userId });
  
      if (!incomeDoc) {
        return res.status(404).json({ message: 'Income document not found for this user' });
      }

      const incomeEntry = incomeDoc.incomes.id(incomeId); 
      if (!incomeEntry) {
        return res.status(404).json({ message: 'Income not found' });
      }
  
      Object.assign(incomeEntry, updateFields);
  
      await incomeDoc.save();
  
      return res.status(200).json({ message: 'Income updated successfully', data: incomeEntry });
    } catch (error) {
      return res.status(500).json({ message: 'Error updating income', error: error.message });
    }
  };