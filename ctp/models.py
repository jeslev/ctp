from transformers import BertForSequenceClassification, RobertaForSequenceClassification
from transformers import BertPreTrainedModel, BertModel
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch


class BertHierarchySeqClasification(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.classifier = nn.Linear(config.hidden_size*2, config.num_labels) # for concat
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) # for sum

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        Code copy from https://huggingface.co/transformers/v3.0.2/_modules/transformers/modeling_roberta.html#RobertaForSequenceClassification
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1] # cls token

        # Get hierarchy
        h_outputs= self.hierarchy(input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        cls_hierarchy = h_outputs[1]

        pooled_output = self.dropout(pooled_output)
        #print(cls_hierarchy.shape)
        #print(pooled_output.shape)
        
        #new_output = torch.cat((pooled_output, cls_hierarchy), 1) # for concat
        new_output = pooled_output + cls_hierarchy # for sum
        #logits = self.classifier(pooled_output)
        logits = self.classifier(new_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
