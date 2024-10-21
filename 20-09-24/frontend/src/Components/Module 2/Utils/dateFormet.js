import moment from "moment";

export const dateFormet = (date) => {
    return moment(date).format('DD/MM/YYYY')
}